//本案例对应数据库6个表，因此写了六个请求数据的函数。

//表名称smokefire, falldown

window.falldown_data;

function mysql_read_falldown() {   
    b=$.ajax({
    type:'get', //请求方式
    url:'http://192.168.194.94:3000/falldown',  //请求地址,需要根据自己服务器ip地址来替换
    success:function(result) {  //请求成功以后函数被调用  response为服务器返回数据 该方法内部会将json字符串转为json对象
        console.log(result)
        window.falldown_data = result;
        var myDate = new Date();
        // myDate.getTime();
        // console.log(myDate.getTime());
        // console.log(myDate.getTime()/1000);
        // console.log(result[0].time);
        console.log(myDate.getTime()/1000-result[0].time);
        fall_warning = document.getElementById("falldown");
        if(myDate.getTime()/1000-result[0].time<2000){
            fall_warning.style="color:rgb(253, 88, 88)";
            fall_warning.innerText="告警";
            //console.log(result[0].raw)
            //点击触发事件
            fall_warning.onclick = function () {
                //alert("警告");  //替换文本
                var img = document.getElementById("outimg");
                img.src=result[0].raw;
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
setTimeout("mysql_read_falldown()", 10000);
}
mysql_read_falldown();