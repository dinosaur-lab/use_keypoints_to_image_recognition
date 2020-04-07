import cv2 as cv
import pickle


def get_sources():
    """Get source's features from file

        Returns:
            sources(list): source's keypoints, descriptors,and img
        """
    # 特徴点情報をファイルから取得
    with open("sources_data.pickle", mode="rb") as f:
        sources = pickle.load(f)

    for n in sources:
        items = sources[n]
        # keypointsをbytesからlistに直す
        list(map(list, items["keypoint"]))
        # keypointsを元の構造に復元
        keypoints = []
        for p in items["keypoint"]:
            temp = cv.KeyPoint(
                x=p[0][0],
                y=p[0][1],
                _size=p[1],
                _angle=p[2],
                _response=p[3],
                _octave=p[4],
                _class_id=p[5],
            )
            keypoints.append(temp)
        items["keypoint"] = keypoints

    return sources


matcher = cv.BFMatcher()

# ターゲット画像読み込み
target_img = cv.imread("images/target_girl.jpg")
# 特徴量取得
akaze = cv.AKAZE_create()
target_kp, target_des = akaze.detectAndCompute(target_img, None)
# ソース画像の特徴点情報をファイルから読み込み
sources = get_sources()

for n in sources:
    source = sources[n]
    source_img = cv.imread("images/" + source["src"])
    matches = matcher.knnMatch(source["descriptor"], target_des, k=2)
    # データを間引きする
    ratio = 0.5
    matched_keypoints = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            matched_keypoints.append([m])

    # 任意の閾値よりgoodが多い場合結果画像を出力
    if len(matched_keypoints) > 20:
        out = cv.drawMatchesKnn(
            source_img,
            source["keypoint"],
            target_img,
            target_kp,
            matched_keypoints,
            None,
            flags=2,
        )

cv.imwrite("images/result.jpg", out)
cv.waitKey()
