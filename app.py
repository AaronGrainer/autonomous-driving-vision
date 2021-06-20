import streamlit as st
import cv2

from config import global_config as gc
from lane_detection.detect import LaneDetection


if __name__ == '__main__':
    vision_select = st.sidebar.selectbox(
        'Which Vision Detection would you like to use?',
        ('Lane Detection', 'Semantic Segmentation')
    )

    if vision_select == 'Lane Detection':
        st.title('Lane Detection using Ultra Fast Structure-aware Deep Lane Detection')

        stframe = st.empty()

        lane_detection = LaneDetection(load_dataloader=False)

        cap = cv2.VideoCapture(gc.lane_detection_video)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                lane_detection.detect(frame)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame,
                              caption='Lanes Detected',
                              use_column_width=True,
                              channels='BGR')
            else:
                break
        cap.release()
        





