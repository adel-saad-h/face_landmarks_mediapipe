import cv2
import numpy as np
import mediapipe as mp
from matplotlib import pyplot as plt
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# def draw_landmarks_on_image(rgb_image, detection_result):
#     face_landmarks_list = detection_result.face_landmarks
#     annotated_image = np.copy(rgb_image)
#
#     # Loop through the detected faces to visualize.
#     for idx in range(len(face_landmarks_list)):
#         face_landmarks = face_landmarks_list[idx]
#
#         # Draw the face landmarks.
#         face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
#         face_landmarks_proto.landmark.extend([
#             landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
#         ])
#
#         mp.solutions.drawing_utils.draw_landmarks(
#             image=annotated_image,
#             landmark_list=face_landmarks_proto,
#             connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
#             landmark_drawing_spec=None,
#             connection_drawing_spec=mp.solutions.drawing_styles
#             .get_default_face_mesh_tesselation_style())
#         mp.solutions.drawing_utils.draw_landmarks(
#             image=annotated_image,
#             landmark_list=face_landmarks_proto,
#             connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
#             landmark_drawing_spec=None,
#             connection_drawing_spec=mp.solutions.drawing_styles
#             .get_default_face_mesh_contours_style())
#         mp.solutions.drawing_utils.draw_landmarks(
#             image=annotated_image,
#             landmark_list=face_landmarks_proto,
#             connections=mp.solutions.face_mesh.FACEMESH_IRISES,
#             landmark_drawing_spec=None,
#             connection_drawing_spec=mp.solutions.drawing_styles
#             .get_default_face_mesh_iris_connections_style())
#
#     return annotated_image
#
#
# def plot_face_blendshapes_bar_graph(face_blendshapes):
#     # Extract the face blendshapes category names and scores.
#     face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
#     face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
#     # The blendshapes are ordered in decreasing score value.
#     face_blendshapes_ranks = range(len(face_blendshapes_names))
#
#     fig, ax = plt.subplots(figsize=(12, 12))
#     bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
#     ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
#     ax.invert_yaxis()
#
#     # Label each bar with values
#     for score, patch in zip(face_blendshapes_scores, bar.patches):
#         plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")
#
#     ax.set_xlabel('Score')
#     ax.set_title("Face Blendshapes")
#     plt.tight_layout()
#     plt.show()
#
#
# model_path = 'face_landmarker.task'
# BaseOptions = mp.tasks.BaseOptions
# FaceLandmarker = mp.tasks.vision.FaceLandmarker
# FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
# VisionRunningMode = mp.tasks.vision.RunningMode
#
# options = FaceLandmarkerOptions(
#     base_options=BaseOptions(model_asset_path=model_path),
#     running_mode=VisionRunningMode.IMAGE,
#     output_face_blendshapes=True,
#     output_facial_transformation_matrixes=True
#
# )
# img = cv2.imread("face.jpg")
# numpy_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)
# with FaceLandmarker.create_from_options(options) as landmarker:
#     face_landmarker_result = landmarker.detect(mp_image)
#     print(face_landmarker_result.face_landmarks)
#     print(face_landmarker_result.face_blendshapes)
#     print(face_landmarker_result.facial_transformation_matrixes[0])
#
#     for c in face_landmarker_result.face_blendshapes[0]:
#         print(c.category_name)
#     annotated_image = draw_landmarks_on_image(numpy_image, face_landmarker_result)
#     # cv2.imshow("I", annotated_image)
#     # cv2.waitKey(0)
#     # plot_face_blendshapes_bar_graph(face_landmarker_result.face_blendshapes[0])
#######################################################################################

cap = cv2.VideoCapture("f01.mp4")
mp_mesh = mp.solutions.face_mesh
mesh = mp_mesh.FaceMesh(max_num_faces=1)

num = 0
mp_draw = mp.solutions.drawing_utils
draw_spec = mp_draw.DrawingSpec(thickness=1, circle_radius=2)
while True:
    num += 1
    success, img = cap.read()
    img2rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mesh.process(img2rgb)
    print(results.multi_face_landmarks)
    # print(mp_mesh.FACEMESH_IRISES)
    # print(mp_mesh.FACEMESH_LEFT_IRIS)
    # print(mp_mesh.FACEMESH_CONTOURS)

    if results.multi_face_landmarks:
        for mesh_lms in results.multi_face_landmarks:
            # print(mesh_lms)
            # mp_draw.draw_landmarks(img, mesh_lms, mp_mesh.FACEMESH_CONTOURS, draw_spec, draw_spec)
            lm_list = []
            h, w, c = img.shape
            for id, lm in enumerate(mesh_lms.landmark):
                print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                lm_list.append([cx, cy])

            # # red part
            # # lt_point = lm_list[162]
            # # br_point = lm_list[345]
            # # cv2.rectangle(img,lt_point,br_point,(255,0,0),1)
            # part_img = img[lm_list[162][1]:lm_list[116][1], lm_list[162][0]:lm_list[389][0]]
            #
            # hsvImg = cv2.cvtColor(part_img, cv2.COLOR_BGR2HSV)
            #
            # # hue 0 red --> 1
            # hsvImg[..., 0] = hsvImg[..., 0] * 0
            #
            # # saturation 0 shades of gray --> 1 no white component in the color
            # hsvImg[..., 1] = hsvImg[..., 1] * 1
            #
            # # value  0 dark --> 1 brighter
            # hsvImg[..., 2] = hsvImg[..., 2] * 1
            # out_img = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)
            #
            # img[lm_list[162][1]:lm_list[116][1], lm_list[162][0]:lm_list[389][0]] = out_img

            # left eye
            # distance between point
            dist_x_left_eye = lm_list[386][0] - lm_list[385][0]
            dist_y_left_eye = lm_list[380][1] - lm_list[385][1]
            center_left_eye = (int(lm_list[380][0] + dist_x_left_eye), int(lm_list[380][1] - dist_y_left_eye / 2))
            radius_left_eye = int(dist_y_left_eye / 2)
            # draw the eye
            cv2.circle(img, center_left_eye, radius_left_eye, (0, 0, 166), -1)
            cv2.circle(img, center_left_eye, int(radius_left_eye / 3), (0, 0, 189), -1)
            cv2.circle(img, center_left_eye, int(radius_left_eye / 4), (0, 0, 203), -1)
            cv2.circle(img, center_left_eye, int(radius_left_eye / 5), (0, 0, 266), -1)
            cv2.circle(img, center_left_eye, int(radius_left_eye / 6), (91, 91, 255), -1)

            # #  right eye
            # # distance between point
            # dist_x_right_eye = lm_list[159][0] - lm_list[159][0]
            # dist_y_right_eye = lm_list[145][1] - lm_list[159][1]
            # center_right_eye = (
            # int(lm_list[145][0] + dist_x_right_eye / 2), int(lm_list[145][1] - dist_y_right_eye / 2))
            # radius_right_eye = int(dist_y_right_eye / 2)
            # # draw the eye
            # # we used radius_left_eye to be the same radius
            # cv2.circle(img, center_right_eye, radius_right_eye, (0, 0, 166), -1)
            # cv2.circle(img, center_right_eye, int(radius_right_eye / 3), (0, 0, 189), -1)
            # cv2.circle(img, center_right_eye, int(radius_right_eye / 4), (0, 0, 203), -1)
            # cv2.circle(img, center_right_eye, int(radius_right_eye / 5), (0, 0, 226), -1)
            # cv2.circle(img, center_right_eye, int(radius_right_eye / 6), (91, 91, 255), -1)

    cv2.namedWindow("IMAGE", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("IMAGE", 1280, 720)
    cv2.imshow("IMAGE", img)
    cv2.imwrite(f"output/image{num}.png", img)
    cv2.waitKey(1)
