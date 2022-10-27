from tqdm import tqdm
import pandas as pd
import numpy as np
import joblib

from sklearn.decomposition import PCA
import umap
import hdbscan

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC

def confusion_matrix(predict, GT):
    length = len(GT)
    classes = len(np.unique(GT))
    matrix = np.zeros((classes,classes))
    for i in range(length):
        matrix[int(predict[i])][int(GT[i])] += 1
    for i in range(classes):
        print(matrix[i])

def bsoidfeat(filename, landmarknum, framerate, savename=None):
    #read csv
    pose_chosen = np.arange(landmarknum*3)
    csv_raw = pd.read_csv(filename, low_memory=False)
    csv_data, _ = adp_filt(csv_raw, pose_chosen)
    #extract feature from points
    feats = bsoid_extract([csv_data], framerate)
    if savename:
        joblib.dump(feats[0], savename)
    return feats[0]

def embedfeat(feat, num_dimensions=None, savename=None):
    if not num_dimensions:
        pca = PCA()
        pca.fit(feat)
        num_dimensions = np.argwhere(np.cumsum(pca.explained_variance_ratio_) >= 0.7)[0][0] + 1
    sampled_input_feats = feat[np.random.choice(feat.shape[0], feat.shape[0], replace=False)]
    learned_embeddings = umap.UMAP(n_neighbors=60, n_components=num_dimensions, min_dist=0.0, random_state=42).fit(sampled_input_feats)
    embeddings = learned_embeddings.embedding_
    if savename:
        joblib.dump(learned_embeddings, savename)
    return learned_embeddings, embeddings

def motion_cluster(embeddings, min_c):
    min_c = 2
    print("min cluster size: ", int(round(min_c * 0.01 * embeddings.shape[0])))
    learned_hierarchy = hdbscan.HDBSCAN(
                        prediction_data=True, min_cluster_size=int(round(min_c * 0.01 * embeddings.shape[0])),
                        min_samples=1).fit(embeddings)
    labels = learned_hierarchy.labels_
    assign_prob = hdbscan.all_points_membership_vectors(learned_hierarchy)
    assignments = np.argmax(assign_prob, axis=1)
    print("motions num: ", len(np.unique(assignments)))
    return assignments

def motion_clf(x, y, test_part=0.1, score=False, savename=None):
    if test_part:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    else:
        x_train, y_train = x, y
    validate_clf = SVC(kernel='rbf', C=100)#RandomForestClassifier(random_state=0)
    validate_clf.fit(x_train, y_train)
    clf = SVC(kernel='rbf', C=100)#RandomForestClassifier(random_state=42)
    clf.fit(x, y)
    if(score):
        print(cross_val_score(validate_clf, x_test, y_test, cv=5, n_jobs=-1))
    if savename:
        joblib.dump(clf, savename)
    return clf

def motion_predict(feat, embeder, clf):
    test_embedding = embeder.transform(feat)
    labels = clf.predict(test_embedding)
    return labels


#################### BSOID migrate functions #######################################################################
import itertools
import math
import numpy as np
from sklearn.preprocessing import StandardScaler


def boxcar_center(a, n):
    a1 = pd.Series(a)
    moving_avg = np.array(a1.rolling(window=n, min_periods=1, center=True).mean())

    return moving_avg

def bsoid_extract(data, fps, frameshift=False):
    """
    Extracts features based on (x,y) positions
    :param data: list, csv data
    :param fps: scalar, input for camera frame-rate
    :return f_10fps: 2D array, extracted features
    """
    win_len = np.int(np.round(0.05 / (1 / fps)) * 2 - 1)
    feats = []
    for m in range(len(data)):
        dataRange = len(data[m])
        dxy_r = []
        dis_r = []
        for r in range(dataRange):
            if r < dataRange - 1:
                dis = []
                for c in range(0, data[m].shape[1], 2):
                    dis.append(np.linalg.norm(data[m][r + 1, c:c + 2] - data[m][r, c:c + 2]))
                dis_r.append(dis)
            dxy = []
            for i, j in itertools.combinations(range(0, data[m].shape[1], 2), 2):
                dxy.append(data[m][r, i:i + 2] - data[m][r, j:j + 2])
            dxy_r.append(dxy)
        dis_r = np.array(dis_r)
        dxy_r = np.array(dxy_r)
        dis_smth = []
        dxy_eu = np.zeros([dataRange, dxy_r.shape[1]])
        ang = np.zeros([dataRange - 1, dxy_r.shape[1]])
        dxy_smth = []
        ang_smth = []
        for l in range(dis_r.shape[1]):
            dis_smth.append(boxcar_center(dis_r[:, l], win_len))
        for k in range(dxy_r.shape[1]):
            for kk in range(dataRange):
                dxy_eu[kk, k] = np.linalg.norm(dxy_r[kk, k, :])
                if kk < dataRange - 1:
                    b_3d = np.hstack([dxy_r[kk + 1, k, :], 0])
                    a_3d = np.hstack([dxy_r[kk, k, :], 0])
                    c = np.cross(b_3d, a_3d)
                    ang[kk, k] = np.dot(np.dot(np.sign(c[2]), 180) / np.pi,
                                        math.atan2(np.linalg.norm(c),
                                                   np.dot(dxy_r[kk, k, :], dxy_r[kk + 1, k, :])))
            dxy_smth.append(boxcar_center(dxy_eu[:, k], win_len))
            ang_smth.append(boxcar_center(ang[:, k], win_len))
        dis_smth = np.array(dis_smth)
        dxy_smth = np.array(dxy_smth)
        ang_smth = np.array(ang_smth)
        feats.append(np.vstack((dxy_smth[:, 1:], ang_smth, dis_smth)))

    f_10fps = []
    for n in range(0, len(feats)):
        feats1 = np.zeros(len(data[n]))
        for s in range(math.floor(fps / 10)):
            for k in range(round(fps / 10) + s, len(feats[n][0]), round(fps / 10)):
                if k > round(fps / 10) + s:
                    feats1 = np.concatenate((feats1.reshape(feats1.shape[0], feats1.shape[1]),
                                                np.hstack((np.mean((feats[n][0:dxy_smth.shape[0],
                                                                    range(k - round(fps / 10), k)]), axis=1),
                                                        np.sum((feats[n][dxy_smth.shape[0]:feats[n].shape[0],
                                                                range(k - round(fps / 10), k)]),
                                                                axis=1))).reshape(len(feats[0]), 1)), axis=1)
                else:
                    feats1 = np.hstack((np.mean((feats[n][0:dxy_smth.shape[0], range(k - round(fps / 10), k)]),
                                                axis=1),
                                        np.sum((feats[n][dxy_smth.shape[0]:feats[n].shape[0],
                                                range(k - round(fps / 10), k)]), axis=1))).reshape(len(feats[0]), 1)
            scaler = StandardScaler()
            scaler.fit(feats1.T)
            scaled_feats1 = scaler.transform(feats1.T)
            f_10fps.append(scaled_feats1)
            if not frameshift:
                break # no frame shift
    return f_10fps


def bsoid_predict(feats, clf):
    """
    :param feats: list, multiple feats (original feature space)
    :param clf: Obj, MLP classifier
    :return nonfs_labels: list, label/100ms
    """
    labels_fslow = []
    for i in range(0, len(feats)):
        labels = clf.predict(feats[i])
        labels_fslow.append(labels)
    return labels_fslow


def adp_filt(currdf: object, pose):
    lIndex = []
    xIndex = []
    yIndex = []
    currdf = np.array(currdf[1:])
    for header in pose:
        if currdf[0][header + 1] == "likelihood":
            lIndex.append(header)
        elif currdf[0][header + 1] == "x":
            xIndex.append(header)
        elif currdf[0][header + 1] == "y":
            yIndex.append(header)
    curr_df1 = currdf[:, 1:]
    datax = curr_df1[1:, np.array(xIndex)]
    datay = curr_df1[1:, np.array(yIndex)]
    data_lh = curr_df1[1:, np.array(lIndex)]
    currdf_filt = np.zeros((datax.shape[0], (datax.shape[1]) * 2))
    perc_rect = []
    for i in range(data_lh.shape[1]):
        perc_rect.append(0)
    for x in tqdm(range(data_lh.shape[1])):
        a, b = np.histogram(data_lh[1:, x].astype(np.float))
        rise_a = np.where(np.diff(a) >= 0)
        if rise_a[0][0] > 1:
            llh = b[rise_a[0][0]]
        else:
            llh = b[rise_a[0][1]]
        data_lh_float = data_lh[:, x].astype(np.float)
        perc_rect[x] = np.sum(data_lh_float < llh) / data_lh.shape[0]
        currdf_filt[0, (2 * x):(2 * x + 2)] = np.hstack([datax[0, x], datay[0, x]])
        for i in range(1, data_lh.shape[0]):
            if data_lh_float[i] < llh:
                currdf_filt[i, (2 * x):(2 * x + 2)] = currdf_filt[i - 1, (2 * x):(2 * x + 2)]
            else:
                currdf_filt[i, (2 * x):(2 * x + 2)] = np.hstack([datax[i, x], datay[i, x]])
    currdf_filt = np.array(currdf_filt)
    currdf_filt = currdf_filt.astype(np.float)
    return currdf_filt, perc_rect
#########################################################################################################################