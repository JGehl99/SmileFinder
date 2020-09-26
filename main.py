from flask import Flask, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app)


@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('webcam.html')


@socketio.on('image')
def image(img_data):

    model = Model()

    # String buffer
    strbuf = StringIO()
    strbuf.write(img_data)

    # Decode the data and convert it into an image
    img_bytes = io.BytesIO(base64.b64decode(img_data))
    img = Image.open(img_bytes)

    # Convert the image to a type that is supported by opencv
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Process and interpred image with Model
    frame = imutils.resize(frame, width=700)
    frame = cv2.flip(frame, 1)

    frame = model.predict(frame)

    imgencode = cv2.imencode('.jpg', frame)[1]

    # base64 encode
    b64_string = base64.b64encode(imgencode).decode('utf-8')
    b64_src = 'data:image/jpg;base64,'
    b64_string = b64_src + b64_string

    # emit the frame back
    emit('response_back', b64_string)


if __name__ == '__main__':
    # defining server ip address and port
    app.run(host='0.0.0.0', port='5000', debug=True)
