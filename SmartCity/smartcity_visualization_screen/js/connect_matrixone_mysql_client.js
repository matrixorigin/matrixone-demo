function mysql_read() {   
    b=$.ajax({
    type:'get', //请求方式
    url:'http://10.112.192.186:3000/api',  //请求地址,需要根据自己服务器ip地址来替换
    success:function(result) {  //请求成功以后函数被调用  response为服务器返回数据 该方法内部会将json字符串转为json对象
        console.log(result)
        a=result.名字;
        b=result.性别;
        c=result.年龄;
            },
    error:function (xhr){  //请求失败后调用
    console.log(xhr);
            }
          });
setTimeout("mysql_read()", 10000);
}
mysql_read();