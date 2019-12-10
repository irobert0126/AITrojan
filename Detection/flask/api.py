import os

template = '<div class="row"> <p>Episode %s</p> <br/> <img src="%s" width="1500px" /></div><br /><br />'

def status(flask):
    data = ""
    if flask.request.method == "GET":
        folder = flask.request.args.get('folder')
    return flask.jsonify(data)

def display(flask):
    data = ""
    if flask.request.method == "GET":
        e = flask.request.args.get('e')
        dir = '/static/result/imgs/adv_step%s_all.png' % e
        print(dir)
        if os.path.exists(dir[1:]):
            return template % (e, dir)
        return ""