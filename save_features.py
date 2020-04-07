import cv2 as cv
import pickle

SOURCE_FILES = [
    "youngwoman_37.jpg",
    "youngwoman_38.jpg",
    "youngwoman_39.jpg",
    "youngwoman_40.jpg",
    "youngwoman_41.jpg",
    "youngwoman_42.jpg",
    "youngwoman_43.jpg",
    "youngwoman_44.jpg",
    "youngwoman_45.jpg",
    "youngwoman_46.jpg",
    "youngwoman_47.jpg",
    "youngwoman_48.jpg",
]


def get_features(img_file_name):
    """Get features of master images

        Args:
            img_file_name(list): Master image

        Returns:
            keypoints, descriptors, img

    """
    img = cv.imread("images/" + img_file_name)

    # 特徴点情報抽出
    akaze = cv.AKAZE_create()
    kp, des = akaze.detectAndCompute(img, None)

    features = [kp, des]

    return features


sources = {}
for item in SOURCE_FILES:
    features = get_features(item)
    # keypointをlist化
    keypoints = []
    for p in features[0]:
        temp = (p.pt, p.size, p.angle, p.response, p.octave, p.class_id)
        keypoints.append(temp)

    # keypointsをbytesに変換
    map(bytes, keypoints)
    # 特徴点情報を辞書化
    sources[item] = {
        "src": item,
        "keypoint": keypoints,
        "descriptor": features[1],
    }

# 特徴点情報をファイルに書き込み
with open("sources_data.pickle", mode="wb") as f:
    pickle.dump(sources, f)
