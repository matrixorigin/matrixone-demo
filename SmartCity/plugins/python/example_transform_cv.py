#!/usr/bin/python3

#
# Demonstrates how to run a Python multi model dection on a video with matrixone database.
#
# Usage:
#   GST_PLUGIN_PATH=$PWD/..:$GST_PLUGIN_PATH \
#   gst-launch-1.0 videotestsrc ! example_transform_cv ! autovideosink
#
# See also https://mathieuduponchelle.github.io/2018-02-15-Python-Elements-2.html
#

import sys
sys.path.append("/root/nxytest/plugins/python/")
from face.test_img_pravega import test_face
from face.utils.load_models import load_face_models
from falling.test_img_pravega import test_falldown, load_falldown_model
from manhole.test_img_pravega import test_manhole, load_manhole_model
from multiobject.test_img_pravega import test_multiobject, load_multiobject_model
from smokefire.test_img_pravega import test_smokefire, load_smokefire_model
from vehicle.test_img_pravega import test_vehicle, load_vehicle_model

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
gi.require_version('GstVideo', '1.0')
from gi.repository import Gst, GObject, GstBase, GstVideo
import gstreamer.utils as utils
import pymysql
import numpy as np

FIXED_CAPS = Gst.Caps.from_string('video/x-raw,format=RGB')

class ExampleTransform(GstBase.BaseTransform):
    __gstmetadata__ = (
        'example_python_transform_cv',
        'Transform',
        'Demonstrates how to run a simple Python cv2 transformation on a video',
        'Ryan')
        
    _srctemplate = Gst.PadTemplate.new('src', Gst.PadDirection.SRC,
                                       Gst.PadPresence.ALWAYS,
                                       FIXED_CAPS)

    _sinktemplate = Gst.PadTemplate.new('sink', Gst.PadDirection.SINK,
                                        Gst.PadPresence.ALWAYS,
                                        FIXED_CAPS)

    __gsttemplates__ = (_srctemplate, _sinktemplate)
   
    
    def __init__(self):
        super().__init__()
        self.iscuda=False

        # Face model
        exp_file = "/root/nxytest/plugins/python/face/YOLOX/exps/example/custom/yolox_tiny.py"
        ckpt = "/root/nxytest/plugins/python/face/YOLOX/YOLOX_outputs/yolox-tiny/best_ckpt.pth"
        BACKBONE_RESUME_ROOT = '/root/nxytest/plugins/python/face/trained_models/backbone_ir50_asia.pth'
        classifier_root = '/root/nxytest/plugins/python/face/trained_models/CLASSIFIER-B0-224-Epoch_170.pth'
        self.face_predictor, self.BACKBONE, self.classifier = load_face_models(exp_file, ckpt, BACKBONE_RESUME_ROOT, classifier_root)
        # Falldown model
        self.counter=0
        self.fall_counter=0
        self.f_c = np.zeros(20)
        self.bbo_cen = np.zeros(20)
        self.fall_judge = np.zeros(20)
        self.fall_up_bound = np.zeros(20)
        self.detect_fall = np.zeros(20)
        self.bbo_all_num = -1
        exp_file="/root/nxytest/plugins/python/falling/yolox_voc_s_down.py"
        self.falling_predictor=load_falldown_model(exp_file,self.iscuda)
        # Manhole model
        self.well_counter = 0#单位：个
        fps=30
        self.last_detect = -5*fps#单位：帧
        self.judge = 0
        exp_file="/root/nxytest/plugins/python/manhole/yolox_voc_s_down.py"
        self.manhole_predictor = load_manhole_model(exp_file, self.iscuda)
        # Multiobject model
        self.assets_counter=0
        self.person_counter=0
        self.last_assets_detect=0
        self.last_person_detect=0
        exp_file = "/root/nxytest/plugins/python/multiobject/yolox_voc_s_p_c.py"
        exp_dir = "/root/nxytest/plugins/python"
        self.multiobject_predictor = load_multiobject_model(exp_file, exp_dir, self.iscuda)
        # Smokefire model
        classfier_root='/root/nxytest/plugins/python/smokefire/model/CLASSIFIER-B0-224-Epoch_122.pth'
        self.smokefire_predictor=load_smokefire_model(classfier_root,self.iscuda)
        # Vehicle model
        exp_file = "/root/nxytest/plugins/python/multiobject/yolox_voc_s_p_c.py"
        self.vehicle_predictor = load_vehicle_model(exp_file, exp_dir, self.iscuda)

       # Initial database
        Gst.info("Database init")
        self.db1=self.create_db_scheme("face")
        self.db2=self.create_db_scheme("falldown")
        self.db3=self.create_db_scheme("manhole")
        self.db4=self.create_db_scheme("multiobject")
        self.db5=self.create_db_scheme("smokefire")
        self.db6=self.create_db_scheme("vehicle")
        # self.db=None

        
        Gst.info("Started.")
    
    def create_db_scheme(self, name):
        # Open a MatrixOne connection
        db = pymysql.connect(host='127.0.0.1',
                    port=6001,
                    user='dump',
                    password='111',
                    database='park')
                    
        # Create db schema(create tables)
        cursor = db.cursor()
        if name=="face":
            sql='create table if not exists face(\
                camera_id int(4) NOT NULL,\
                frame_id int(10) NOT NULL,\
                `time` datetime,\
                `raw` blob,\
                face_x1 int(6),face_y1 int(6),face_x2 int(6),face_y2 int(6),\
                sex_category varchar(10),sex_confidence float,\
                age_number int(4),age_confidence float,\
                mask_state int(2),mask_confidence float,\
                glasses_state int(2),glasses_confidence float);'
        elif name=="falldown":
            sql='create table if not exists falldown(\
                camera_id int(4) NOT NULL,\
                frame_id int(10) NOT NULL,\
                `time` int(10) NOT NULL,\
                `raw` blob,\
                person varchar(5000));'
        elif name=="manhole":
            sql='create table if not exists well(\
                camera_id int(4) NOT NULL,\
                frame_id int(10) NOT NULL,\
                `time` int(10) NOT NULL,\
                `raw` blob,\
                manhole_cover varchar(5000));'
        elif name=="multiobject":
            sql='create table if not exists multiobject(\
                camera_id int(4) NOT NULL,\
                frame_id int(10) NOT NULL,\
                `time` int(10) NOT NULL,\
                raw blob,\
                person varchar(5000),\
                environment varchar(1000));'
        elif name=="smokefire":
            sql='create table if not exists smokefire(\
                camera_id int(4) NOT NULL,\
                frame_id int(10) NOT NULL,\
                `time` datetime,\
                raw blob,\
                environment varchar(500));'
        elif name =="vehicle":
            sql='create table if not exists vehicle(\
                camera_id int(4) NOT NULL,\
                frame_id int(10) NOT NULL,\
                `time` varchar(20) NOT NULL,\
                raw blob,\
                vehicle varchar(5000),\
                license_plate varchar(5000));'
        # sql = face_sql + falldown_sql + manhole_sql + multiobject_sql + smokefire_sql + vehicle_sql
        cursor.execute(sql)
        db.commit()
        return db

    def update_counter(self):
        self.counter  += 1

    def do_set_caps(self, incaps, outcaps):
        struct = incaps.get_structure(0)
        self.width = struct.get_int("width").value
        self.height = struct.get_int("height").value
        Gst.info("width=%d, height=%d" % (self.width, self.height))
        return True

    def do_transform_ip(self, buffer: Gst.Buffer) -> Gst.FlowReturn:
        try:
            self.update_counter()
            image = utils.gst_buffer_with_caps_to_ndarray(buffer, self.sinkpad.get_current_caps())
            # face
            test_face(image,self.face_predictor,self.counter,self.BACKBONE,self.classifier,self.iscuda,self.db1) 
            # falldown
            self.fall_counter,self.f_c,self.bbo_cen,self.fall_judge,self.fall_up_bound,self.detect_fall,self.bbo_all_num= \
                test_falldown(image,self.falling_predictor,self.counter,self.fall_counter,self.f_c,self.bbo_cen,self.fall_judge,self.fall_up_bound,self.detect_fall,self.bbo_all_num,self.iscuda,self.db2)
            # manhole
            self.well_counter,self.last_detect,self.judge= \
                test_manhole(image,self.manhole_predictor,self.counter,self.well_counter,self.last_detect,self.judge,self.iscuda,self.db3)
            # multiobject
            self.assets_counter, self.person_counter, self.last_assets_detect, self.last_person_detect = \
                test_multiobject(image, self.multiobject_predictor, self.counter, self.assets_counter,
                                self.person_counter, self.last_assets_detect, self.last_person_detect, self.iscuda, self.db4)
            # smokefire
            test_smokefire(image, self.smokefire_predictor,self.counter,self.iscuda,self.db5)
            # vehicle
            test_vehicle(image,self.vehicle_predictor,self.iscuda,self.db6)

        except Exception as e:
            Gst.error("Mapping error: %s" % e)
            return Gst.FlowReturn.ERROR

        return Gst.FlowReturn.OK
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db.close()
        Gst.info(f"Shutdown {self}")

GObject.type_register(ExampleTransform)
__gstelementfactory__ = ("example_transform_cv", Gst.Rank.NONE, ExampleTransform)
