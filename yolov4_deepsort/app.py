from flask import Flask, request, jsonify, render_template
from object_track import Object_tracker

app = Flask(__name__,template_folder='templates')

@app.route('/', methods=['GET','POST'])
def count_objects():
    if request.method == 'GET':
        return render_template('upload.html')
    video_url = request.files['video']
    video_path = 'videos/' + video_url.filename
    video_url.save(video_path)
    # Download video from URL and save to local file
    # Call object tracker function with local video file path
    # Get the count of objects detected by the tracker
    objects = Object_tracker(video_path, 416, 0.45, 0.50, count=True)
    objects_count = objects.process()
    return objects_count

if __name__ == '__main__':
    app.run(debug=True,port=8000)
