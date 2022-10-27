//本案例对应数据库6个表，因此写了六个请求数据的函数。每个函数中url代表server端地址，使用时需要根据实际情况进行修改。

//mysql_read_falldown：摔倒数据读取函数，读取falldown表中最近一条数据，如果该数据与当前时间间隔小于2000秒则显示告警，否则显示正常。

function mysql_read_falldown() {   
    b=$.ajax({
    type:'get', //请求方式
    //MatrixONE MySQL地址
    url:'http://10.112.89.158:3000/falldown',  //请求地址,需要根据自己服务器ip地址来替换
    //MySQL地址
    //url:'http://192.168.194.94:3001/falldown',
    success:function(result) {  //请求成功以后函数被调用  response为服务器返回数据 该方法内部会将json字符串转为json对象
        console.log('falldown:',result)
        //window.falldown_data = result;
        var myDate = new Date();
        // myDate.getTime();
        // console.log(myDate.getTime());
        // console.log(myDate.getTime()/1000);
        // console.log(result[0].time);
        //console.log(myDate.getTime()/1000-result[0].time);
        fall_warning = document.getElementById("falldown");
        //if(myDate.getTime()/1000-result[0].time<2000){
        if(myDate.getTime()/1000-result[0].time<2000){
            fall_warning.style="color:rgb(253, 88, 88)";
            fall_warning.innerText="告警";
            //console.log(result[0].raw)
            //点击触发事件
            fall_warning.onclick = function () {
                //alert("警告");  //替换文本
                var img = document.getElementById("outimg");
                img.src=result[0].raw; //base64
                //二进制
                // const objectURL = URL.createObjectURL(result[0].raw);
                // img.src=objectURL;

                // const binaryData = [];
                // binaryData.push(result[0].raw);
                // url = window.URL.createObjectURL(new Blob(binaryData,{type:'image/jpg'}));
                // console.log(url)
                // img.src=url;

                // const buffer = new Buffer(result[0].raw, 'binary');
                // img.src = 'data: image/'+ getImageType(fileName) +';base64,' + buffer.toString('base64');
                // console.log(result[0].raw)
                // const base64 = blobToDataURI(result[0].raw);
                // img.src = 'data: image/jpg;base64,' + String(base64);

            }
        }else{
            fall_warning.style="color:rgb(88, 253, 94)";
            fall_warning.innerText="正常";
        }
            },
    error:function (xhr){  //请求失败后调用
    console.log(xhr);
            }
          });
setTimeout("mysql_read_falldown()", 5000);
}
//二进制转base64
// function blobToDataURI(blob, callback) {
//     var reader = new FileReader();
//     reader.readAsDataURL(blob);
//     reader.onload = function (e) {
//         callback(e.target.result);
//     }
//  }

//mysql_read_face：人脸数据读取函数
function mysql_read_face() {   
    b=$.ajax({
    type:'get', //请求方式
    //MatrixONE MySQL地址
    url:'http://10.112.89.158:3000/face',  //请求地址,需要根据自己服务器ip地址来替换
    //MySQL地址
    //url:'http://192.168.194.94:3001/falldown',
    success:function(result) {  //请求成功以后函数被调用  response为服务器返回数据 该方法内部会将json字符串转为json对象
        console.log('face:',result)
        console.log('face_male:',result[0][0]['count(*)'])
        window.male_num = result[0][0]['count(*)'];
        console.log('face_female:',result[1][0]['count(*)'])
        window.female_num = result[1][0]['count(*)'];
        console.log('face_with_mask:',result[2][0]['count(*)'])
        window.with_mask_num = result[2][0]['count(*)'];
        people_num_text = document.getElementById("people_num");
        window.people_num = result[0][0]['count(*)'] + result[1][0]['count(*)'];
        window.without_mask_num = window.people_num-result[2][0]['count(*)'];
        console.log('people_num:',window.people_num);
        people_num_text.innerText=window.people_num;
            },
    error:function (xhr){  //请求失败后调用
    console.log(xhr);
            }
          });
setTimeout("mysql_read_face()", 60000);
}

//mysql_read_well：井盖数据读取函数
function mysql_read_well() {   
    b=$.ajax({
    type:'get', //请求方式
    //MatrixONE MySQL地址
    url:'http://10.112.89.158:3000/well',  //请求地址,需要根据自己服务器ip地址来替换
    //MySQL地址
    //url:'http://192.168.194.94:3001/falldown',
    success:function(result) {  //请求成功以后函数被调用  response为服务器返回数据 该方法内部会将json字符串转为json对象
        console.log('well:',result)
        //window.falldown_data = result;
        var myDate = new Date();
        //console.log(myDate.getTime()/1000-result[0].time);
        well_warning = document.getElementById("well");
        //console.log('well time:',myDate.getTime()/1000)
        //console.log('well time:',result[0].time)
        //console.log('well time:',myDate.getTime()/1000-result[0].time)
        if(myDate.getTime()/1000-result[0].time<2000){
            well_warning.style="color:rgb(253, 88, 88)";
            well_warning.innerText="告警";
            //console.log(result[0].raw)
            //点击触发事件
            well_warning.onclick = function () {
                //alert("警告");  //替换文本
                var img = document.getElementById("center_outimg");
                img.src=result[0].raw; //base64
            }
        }else{
            well_warning.style="color:rgb(88, 253, 94)";
            well_warning.innerText="正常";
        }
            },
    error:function (xhr){  //请求失败后调用
    console.log(xhr);
            }
          });
setTimeout("mysql_read_well()", 60000);
}

//mysql_read_multiobject：多目标数据读取函数
function mysql_read_multiobject() {   
    b=$.ajax({
    type:'get', //请求方式
    //MatrixONE MySQL地址
    url:'http://10.112.89.158:3000/multiobject',  //请求地址,需要根据自己服务器ip地址来替换
    //MySQL地址
    //url:'http://192.168.194.94:3001/falldown',
    success:function(result) {  //请求成功以后函数被调用  response为服务器返回数据 该方法内部会将json字符串转为json对象
        console.log('multiobject:',result)
        window.falldown_data = result;
        var myDate = new Date();
        //console.log(myDate.getTime()/1000-result[0].time);
        hat_warning = document.getElementById("hat");
        //点击触发事件
        hat_warning.onclick = function () {
            //alert("警告");  //替换文本
            console.log('click hat')
            var img = document.getElementById("center_outimg");
            img.src=result[0].raw; //base64
        }
            },
    error:function (xhr){  //请求失败后调用
    console.log(xhr);
            }
          });
setTimeout("mysql_read_multiobject()", 60000);
}

//mysql_read_smokefire：烟火数据读取函数
function mysql_read_smokefire() {   
    b=$.ajax({
    type:'get', //请求方式
    //MatrixONE MySQL地址
    url:'http://10.112.89.158:3000/smokefire',  //请求地址,需要根据自己服务器ip地址来替换
    //MySQL地址
    //url:'http://192.168.194.94:3001/falldown',
    success:function(result) {  //请求成功以后函数被调用  response为服务器返回数据 该方法内部会将json字符串转为json对象
        console.log('smokefire:',result)
        //console.log('smokefire:',result.environment)
        var env = String(result.environment).split(',');
        //console.log('smokefire:',env);
        console.log('fire:',env[0].slice(10,13));
        console.log('smoke:',env[1].slice(11,14));
        //window.falldown_data = result;
        var myDate = new Date();
        //console.log(myDate.getTime()/1000-result[0].time);
        fire_warning = document.getElementById("fire");
        smoke_warning = document.getElementById("smoke");
        if(env[1].slice(11,14)>0.5){
            smoke_warning.style="color:rgb(253, 88, 88)";
            smoke_warning.innerText="告警";
            //console.log(result[0].raw)
            //点击触发事件
            smoke_warning.onclick = function () {
                //alert("警告");  //替换文本
                var img = document.getElementById("center_outimg");
                img.src=result.raw; //base64
            }
        }else{
            smoke_warning.style="color:rgb(88, 253, 94)";
            smoke_warning.innerText="正常";
        }
        if(env[0].slice(10,13)>0.5){
            fire_warning.style="color:rgb(253, 88, 88)";
            fire_warning.innerText="告警";
            //console.log(result[0].raw)
            //点击触发事件
            fire_warning.onclick = function () {
                //alert("警告");  //替换文本
                var img = document.getElementById("center_outimg");
                img.src=result.raw; //base64
            }
        }else{
            fire_warning.style="color:rgb(88, 253, 94)";
            fire_warning.innerText="正常";
        }
        
            },
    error:function (xhr){  //请求失败后调用
    console.log(xhr);
            }
          });
setTimeout("mysql_read_smokefire()", 60000);
}

//mysql_read_vehicle：车辆数据读取函数
function mysql_read_vehicle() {   
    b=$.ajax({
    type:'get', //请求方式
    //MatrixONE MySQL地址
    url:'http://10.112.89.158:3000/vehicle',  //请求地址,需要根据自己服务器ip地址来替换
    //MySQL地址
    //url:'http://192.168.194.94:3001/falldown',
    success:function(result) {  //请求成功以后函数被调用  response为服务器返回数据 该方法内部会将json字符串转为json对象
        console.log('vehicle:',result)
        window.falldown_data = result;
        var myDate = new Date();
        //console.log(myDate.getTime()/1000-result[0].time);
        fall_warning = document.getElementById("falldown");
        if(myDate.getTime()/1000-result[0].time<2000){
            fall_warning.style="color:rgb(253, 88, 88)";
            fall_warning.innerText="告警";
            //console.log(result[0].raw)
            //点击触发事件
            fall_warning.onclick = function () {
                //alert("警告");  //替换文本
                var img = document.getElementById("outimg");
                img.src=result[0].raw; //base64
            }
        }else{
            fall_warning.style="color:rgb(88, 253, 94)";
            fall_warning.innerText="正常";
        }
            },
    error:function (xhr){  //请求失败后调用
    console.log(xhr);
            }
          });
setTimeout("mysql_read_vehicle()", 60000);
}
mysql_read_falldown();
mysql_read_face();
mysql_read_well();
mysql_read_multiobject();
mysql_read_smokefire();
mysql_read_vehicle();