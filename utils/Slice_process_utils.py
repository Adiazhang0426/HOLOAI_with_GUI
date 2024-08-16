"""
@Description: 
@Author: Adiazhang
@Date: 2024-06-04 15:21:27
@LastEditTime: 2024-06-04 17:26:44
@LastEditors: Adiazhang
"""

from filelock import FileLock
import json
import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
from skimage import measure
from multiprocessing import Pool

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from PySide2.QtCore import Signal, QObject, QThread, QRunnable, QThreadPool


def write_json(path, data):
    lock = FileLock(path + ".lock")
    with lock:
        with open(path, "w") as f:
            json.dump(data, f)
        lock.release()


def iou(boxes0, boxes1):

    A = boxes0.shape[0]
    B = boxes1.shape[0]

    xy_max = np.minimum(
        boxes0[:, np.newaxis, 2:].repeat(B, axis=1),
        np.broadcast_to(boxes1[:, 2:], (A, B, 2)),
    )
    xy_min = np.maximum(
        boxes0[:, np.newaxis, :2].repeat(B, axis=1),
        np.broadcast_to(boxes1[:, :2], (A, B, 2)),
    )

    inter = np.clip(xy_max - xy_min, a_min=0, a_max=np.inf)
    inter = inter[:, :, 0] * inter[:, :, 1]

    area_0 = ((boxes0[:, 2] - boxes0[:, 0]) * (boxes0[:, 3] - boxes0[:, 1]))[
        :, np.newaxis
    ].repeat(B, axis=1)
    area_1 = ((boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1]))[
        np.newaxis, :
    ].repeat(A, axis=0)

    return inter / (area_0 + area_1 - inter)


def normalize(a):
    a = np.array(a)
    return (a - np.min(a)) / (np.max(a) - np.min(a))


def preprocess_segmask(img, seg, cluster):

    judge = []
    for i in range(cluster):
        test = np.zeros_like(seg)
        test[seg == i] = 1
        judge.append(np.average(test * img))
    trueid = judge.index(min(judge))
    seg[seg == trueid] = 255
    seg[seg != 255] = 0
    return seg


#
class slice_process_thread:

    def __init__(self, kwargs, patch):
        super().__init__()
        self.kwargs = kwargs
        self.save_run_root = self.kwargs["save_run_root"]
        self.need_cut = self.kwargs["need_cut"]
        self.mergeiou = self.kwargs["mergeiou"]
        self.cropped_imgshape = self.kwargs["cropped_imgshape"]
        self.cal_interval = self.kwargs["cal_interval"]
        self.pp_config = self.kwargs["pp_config"]
        self.patch = patch

        self.write_json_root = r"logging\logging_slice.json"

    def read_txt(self, imgpath, txtpath):
        imglist = os.listdir(os.path.join(imgpath, self.patch))
        txtpath = os.path.join(txtpath, self.patch, "train", "labels")
        txtlist = os.listdir(txtpath)

        allbox = []
        boximgindex = []
        for i in txtlist:
            with open(os.path.join(txtpath, i), "r") as f:
                for line in f.readlines():
                    [xc, yc, w, h] = [float(j) for j in line.strip().split(" ")[1:]]
                    x1, y1, x2, y2 = (
                        xc - 0.5 * w,
                        yc - 0.5 * h,
                        xc + 0.5 * w,
                        yc + 0.5 * h,
                    )
                    allbox.append([x1, y1, x2, y2])
                    boximgindex.append(imglist.index(i[:-4] + ".jpg"))
        return allbox, boximgindex

    def merge_box(self, ioumatrix, boxes, boximgindex):

        ioumatrix = np.triu(ioumatrix)
        ioumatrix[ioumatrix > self.mergeiou] = 1
        ioumatrix[ioumatrix < self.mergeiou] = 0

        allindex = [i for i in range(ioumatrix.shape[0])]
        ioulist = ioumatrix.tolist()
        totalmerge = []
        totalmergeindex = []

        while len(allindex) > 0:
            singlelist = ioulist[allindex[0]]
            singlemerge = []
            singlemergeindex = []
            singlemergeindex.append(boximgindex[allindex[0]])
            singlemerge.append(boxes[allindex[0]])
            allindex.remove(allindex[0])
            for j in range(len(singlelist)):
                if singlelist[j] == 1 and j in allindex:
                    singlemerge.append(boxes[j])
                    singlemergeindex.append(boximgindex[j])
                    allindex.remove(j)
            singlemerge = np.array(singlemerge)
            mx1, my1, mx2, my2 = (
                np.average(singlemerge[:, 0]),
                np.average(singlemerge[:, 1]),
                np.average(singlemerge[:, 2]),
                np.average(singlemerge[:, 3]),
            )
            totalmerge.append(
                [
                    int(mx1 * self.cropped_imgshape[1]),
                    int(my1 * self.cropped_imgshape[0]),
                    int(mx2 * self.cropped_imgshape[1]),
                    int(my2 * self.cropped_imgshape[0]),
                ]
            )
            totalmergeindex.append(singlemergeindex)

        return totalmerge, totalmergeindex

    def z_location_gradient_var_slice(self, rawimgpath, box, index):

        x1, y1, x2, y2 = box
        imglist = os.listdir(os.path.join(rawimgpath, self.patch))
        gv = []

        if len(index) != 1:
            newrange = [
                i
                for i in range(
                    max(0, min(index) - 10), min(len(imglist), max(index) + 10)
                )
            ]
            targetimglist = [
                cv2.imread(os.path.join(rawimgpath, self.patch, imglist[i]), 0)[
                    y1:y2, x1:x2
                ]
                for i in newrange
            ]
            for i in targetimglist:
                sobelx = cv2.Sobel(i, cv2.CV_64F, 1, 0)
                sobely = cv2.Sobel(i, cv2.CV_64F, 0, 1)
                sobelx = cv2.convertScaleAbs(sobelx)
                sobely = cv2.convertScaleAbs(sobely)
                sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
                gv.append(np.var(sobelxy))
            zid = gv.index(max(gv))

            return newrange[zid], targetimglist[zid], targetimglist, zid

        else:
            img = cv2.imread(os.path.join(rawimgpath, self.patch, imglist[index[0]]), 0)

            return index[0], img, -1, -1

    def cluster_segment_2vector(self, img, fuse_vec, cluster=2):

        flatenimg = fuse_vec.reshape(-1, 2)
        kmeans = KMeans(n_clusters=cluster)
        kmeans.fit(flatenimg)
        seg = kmeans.labels_.reshape(fuse_vec.shape[0], fuse_vec.shape[1])
        seg = preprocess_segmask(img, seg, cluster)
        return seg

    def connective_information_extraction(self, seg):
        # particle_information = []
        min_d, max_d, min_eccentricity, max_eccentricity, min_euler, max_euler = (
            self.pp_config["mind"],
            self.pp_config["maxd"],
            self.pp_config["mine"],
            self.pp_config["maxe"],
            self.pp_config["min_eula"],
            self.pp_config["max_eula"],
        )
        labels = measure.label(seg, connectivity=2)
        properties = measure.regionprops(labels)
        xs, ys, ds = [], [], []
        for prop in properties:
            equal_diameter = prop.equivalent_diameter
            eccentricity = prop.eccentricity
            euler_number = prop.euler_number

            if (
                equal_diameter >= min_d
                and equal_diameter <= max_d
                and eccentricity >= min_eccentricity
                and eccentricity <= max_eccentricity
                and euler_number >= min_euler
                and euler_number <= max_euler
            ):
                xs.append(prop.centroid[1])
                ys.append(prop.centroid[0])
                ds.append(equal_diameter)
        if len(xs) > 0:
            id = ds.index(max(ds))
            x, y, d = xs[id], ys[id], ds[id]
            return x, y, d
        else:
            return -1, -1, -1

    def fuse_slice_var_segment(self, imglist, zid, focusimg):

        slicelist = imglist[
            int(max(0, zid - self.cal_interval)) : int(
                min(len(imglist), zid + self.cal_interval)
            )
        ]

        slicelist = np.array(slicelist)
        varmatrix = np.var(slicelist, axis=0)
        varmatrix = normalize(varmatrix)
        focusimg = normalize(focusimg)
        vector = np.concatenate(
            [np.expand_dims(focusimg, axis=-1), np.expand_dims(varmatrix, axis=-1)],
            axis=-1,
        )
        cluster_vec_seg = self.cluster_segment_2vector(focusimg, vector, cluster=2)
        x, y, d = self.connective_information_extraction(cluster_vec_seg)
        if x != -1:
            return x, y, d, cluster_vec_seg
        else:
            return -1, -1, -1, -1

    def delete_repeat(self, partixyz, newxyz):

        np_partixyz = np.array(partixyz)
        np_newxyz = np.array(newxyz)
        contrastxyz = np_partixyz - np_newxyz
        contrastxy = np_partixyz[:, :-1] - np_newxyz[:-1]
        distancexyz = np.min(np.linalg.norm(contrastxyz, axis=-1))
        distancexy = np.min(np.linalg.norm(contrastxy, axis=-1))
        if distancexy == 0:
            return 0
        else:
            if distancexyz < 3.5:
                return 0
            else:
                return 1

    def saveresult(self, resultpath, par):

        with open(os.path.join(resultpath, self.patch + ".txt"), "w") as f:
            for i in par:
                x, y, z, d, bx1, by1, bx2, by2 = (
                    int(i[0]),
                    int(i[1]),
                    int(i[2]),
                    float(i[3]),
                    int(i[4]),
                    int(i[5]),
                    int(i[6]),
                    float(i[7]),
                )
                f.write(
                    "{} {} {} {} {} {} {} {}".format(x, y, z, d, bx1, by1, bx2, by2)
                    + "\n"
                )

    def run(self):
        write_json(self.write_json_root, {"bar_pause": True})
        imgpath = os.path.join(self.save_run_root, "images")
        txtpath = os.path.join(self.save_run_root, "detections")
        resultpath = os.path.join(self.save_run_root, "results", self.patch)
        rawboxes, rawboximgindex = self.read_txt(imgpath, txtpath)
        np_rawboxes = np.array(rawboxes)
        ioumatrix = iou(np_rawboxes, np_rawboxes)
        mergeboxes, mergeindexs = self.merge_box(ioumatrix, rawboxes, rawboximgindex)
        particleinfor = []
        partixy = []
        count = 0
        for mergebox, mergeindex in zip(mergeboxes, mergeindexs):
            bx, by, bx1, by1 = mergebox
            z, resizedimg, targetimglist, zid = self.z_location_gradient_var_slice(
                imgpath, mergebox, mergeindex
            )

            if z != -1 and resizedimg.shape[0] < 100:

                x, y, d, mask = self.fuse_slice_var_segment(
                    targetimglist, zid, resizedimg
                )
                if x != -1:

                    singleinfor = [x + bx, y + by, z, d, bx, by, bx1, by1]
                    if partixy == []:
                        partixy.append([int(x + bx), int(y + by), z])
                        particleinfor.append(singleinfor)
                        cv2.imwrite(
                            os.path.join(resultpath, "raw_{:0>4d}.png".format(count)),
                            resizedimg,
                        )
                        cv2.imwrite(
                            os.path.join(resultpath, "mask_{:0>4d}.png".format(count)),
                            mask,
                        )
                        count += 1
                    if self.delete_repeat(partixy, [int(x + bx), int(y + by), z]) == 1:
                        partixy.append([int(x + bx), int(y + by), z])
                        particleinfor.append(singleinfor)
                        cv2.imwrite(
                            os.path.join(resultpath, "raw_{:0>4d}.png".format(count)),
                            resizedimg,
                        )
                        cv2.imwrite(
                            os.path.join(resultpath, "mask_{:0>4d}.png".format(count)),
                            mask,
                        )
                        count += 1
        self.saveresult(resultpath, particleinfor)

        # self.signals.thread_finish.emit()
        write_json(self.write_json_root, {"bar_update": True})


# 调整参数获取的位置和GUI的传参一致
class Slice_process:
    """
    file structure:
    reconstrcuted_img_root->on_crop
        |-- img_name
            |--original slices
    save_run_root
        |-- img_name
            |-- images
                |-- patch...
                    |--cropped img
            |-- results
                |-- patch...
                    |--ROIs
                    |--masks
                    |--.txt (coordinates based on the cropped image)
                |--.txt (coordinates based on the original image)
            |-- detections
                |-- patch...
                    |--train->labels
                       |--.txt
    """

    def __init__(
        self,
        kwargs,
    ):
        super().__init__()
        self.kwargs = kwargs

        self.task = self.kwargs["task"]
        self.device = self.kwargs["device"]
        self.reconstructed_img_root = self.kwargs["reconstructed_img_root"]
        self.startid = self.kwargs["startid"]
        self.endid = self.kwargs["endid"]
        self.dataset = self.kwargs["dataset"]
        self.model = self.kwargs["model_config_path"]

        self.epoch = self.kwargs["epoch"]
        self.batch_size = self.kwargs["batch_size"]
        self.weights = self.kwargs["weights"]
        self.detection_conf = self.kwargs["detection_conf"]
        self.save_run_root = self.kwargs["save_run_root"]
        self.H_multiple = self.kwargs["H_multiple"]
        self.W_multiple = self.kwargs["W_multiple"]
        self.need_cut = self.kwargs["need_cut"]
        # self.mergeiou = self.kwargs["mergeiou"]
        self.cropped_imgshape = self.kwargs["cropped_imgshape"]
        # self.cal_interval = self.kwargs["cal_interval"]
        self.save_detection_img = self.kwargs["save_detection_img"]

        self.pp_config = self.kwargs["pp_config"]
        if self.task == "infer":
            self.mergeiou = self.pp_config["mergeiou"]
            self.cal_interval = self.pp_config["cal_interval"]
            self.generate_positionlist()
        else:
            self.mergeiou = None
            self.cal_interval = None
        # self.detection_conf=self.pp_config['detection_conf']
        # self.save_detection_img=self.pp_config['save_detection_img']

        self.thread_kwargs = {}
        self.thread_kwargs["save_run_root"] = self.save_run_root
        self.thread_kwargs["need_cut"] = self.need_cut

        self.thread_kwargs["mergeiou"] = self.mergeiou
        self.thread_kwargs["cropped_imgshape"] = self.cropped_imgshape
        self.thread_kwargs["cal_interval"] = self.cal_interval
        self.thread_kwargs["pp_config"] = self.pp_config

        self.count_finished_thread = 0
        self.write_json_root = r"logging\logging_slice.json"

    def generate_positionlist(self):
        self.position_list = []
        for i in range(self.H_multiple):
            for j in range(self.W_multiple):
                self.position_list.append("{}_{}".format(i, j))

    def _crop_img(self):
        no_crop_list = os.listdir(os.path.join(self.reconstructed_img_root, "no_crop"))
        if self.endid == -1:
            endid = len(no_crop_list) + 1
        else:
            endid = 1 + self.endid
        for name in no_crop_list[self.startid : endid]:
            slicepath = os.path.join(self.reconstructed_img_root, "no_crop", name)
            slicelist = os.listdir(slicepath)
            print(name)
            try:
                os.makedirs(os.path.join(self.save_run_root, name, "images"))
                os.makedirs(os.path.join(self.save_run_root, name, "results"))
                os.mkdir(os.path.join(self.save_run_root, name, "detections"))
            except:
                pass
            for slice in slicelist:
                img = cv2.imread(os.path.join(slicepath, slice), 0)

                H, W = img.shape[0], img.shape[1]
                sw, sh = W / self.W_multiple, H / self.H_multiple
                for i in range(self.H_multiple):
                    for j in range(self.W_multiple):
                        cutimg = img[
                            int(i * sh) : int((i + 1) * sh),
                            int(j * sw) : int((j + 1) * sw),
                        ]
                        try:
                            os.mkdir(
                                os.path.join(
                                    self.save_run_root,
                                    name,
                                    "images",
                                    "{}_{}".format(i, j),
                                )
                            )
                            os.mkdir(
                                os.path.join(
                                    self.save_run_root,
                                    name,
                                    "results",
                                    "{}_{}".format(i, j),
                                )
                            )
                            os.makedirs(
                                os.path.join(
                                    self.save_run_root,
                                    name,
                                    "detections",
                                    "{}_{}".format(i, j),
                                )
                            )
                        except:
                            pass
                        cv2.imwrite(
                            os.path.join(
                                self.save_run_root,
                                name,
                                "images",
                                "{}_{}".format(i, j),
                                slice[:-4] + "_{}_{}.jpg".format(i, j),
                            ),
                            cutimg,
                        )

    def YOLO_train(self):
        args = dict(
            model=self.model,
            data=self.dataset,
            epochs=self.epoch,
            device=self.device,
            batch=self.batch_size,
        )
        trainer = DetectionTrainer(overrides=args)
        self.text_print("开始训练YOLO模型")
        trainer.train()

    def YOLO_predict(self, source, save_root):
        args = dict(
            model=self.weights,
            source=source,
            device=self.device,
            save_txt=True,
            conf=self.detection_conf,
            project=save_root,
            save=self.save_detection_img,
        )
        Predictor = DetectionPredictor(overrides=args)
        Predictor.predict_cli()

    def extract_fuse_info(self, root):
        totalparticle = []
        for j in self.position_list:
            xid = int(j.split("_")[1])
            yid = int(j.split("_")[0])
            singletxtpath = os.path.join(root, "results", j, j + ".txt")
            try:
                with open(singletxtpath, "r") as f:
                    for line in f.readlines():
                        x = (
                            float(line.strip().split(" ")[0])
                            + self.cropped_imgshape[1] * xid
                        )
                        y = (
                            float(line.strip().split(" ")[1])
                            + self.cropped_imgshape[0] * yid
                        )
                        z = int(line.strip().split(" ")[2]) + 1
                        d = float(line.strip().split(" ")[3])
                        totalparticle.append([x, y, z, d])
            except:
                pass
        # with open(
        #     os.path.join(root, "results", "_allparticle_imagelevel.txt"), "w"
        # ) as g:
        #     for h in totalparticle:
        #         g.write("{} {} {} {}\n".format(h[0], h[1], h[2], h[3]))
        return totalparticle

    def response_process_finish(self):
        self.count_finished_thread += 1
        # print("finished thread", self.count_finished_thread)
        write_json(self.write_json_root, {"bar_update": self.count_finished_thread})

    def run(self):
        # if self.need_cut:
        #     self._crop_img()
        #     self.text_print.emit("裁剪完成")
        if self.task == "train":
            self.YOLO_train()
            # self.infor_print.emit("训练完成")
            write_json(self.write_json_root, {"msg": "训练完成"})
        else:
            caselist = os.listdir(self.save_run_root)
            all_res = []
            for case in caselist:
                self.processpool = Pool(
                    processes=min(16, self.H_multiple * self.W_multiple)
                )

                write_json(self.write_json_root, {"msg": "处理全息图: " + case})

                case_path = os.path.join(self.save_run_root, case)
                for patch in self.position_list:
                    write_json(self.write_json_root, {"msg": "开始检测: " + case})
                    caseimg_patch_path = os.path.join(case_path, "images", patch)
                    casedetect_patch_path = os.path.join(case_path, "detections", patch)
                    self.YOLO_predict(
                        source=caseimg_patch_path, save_root=casedetect_patch_path
                    )
                write_json(self.write_json_root, {"msg": case + " detection finished"})

                self.thread_kwargs["save_run_root"] = case_path
                with self.processpool as pool:
                    write_json(self.write_json_root, {"bar_clear": True})

                    write_json(
                        self.write_json_root,
                        {"bar_ini": [0, len(self.position_list)]},
                    )
                    for patch in self.position_list:

                        worker = slice_process_thread(self.thread_kwargs, patch)
                        pool.apply(worker.run)
                    # self.processpool.close()
                    # self.processpool.terminate()
                    # self.processpool.join()

                write_json(self.write_json_root, {"msg": case + "定位识别结束"})

                case_res = self.extract_fuse_info(case_path)
                all_res.append(case_res)
                write_json(self.write_json_root, {"msg": case + "运行结束"})

            write_json(self.write_json_root, {"result": all_res})
            write_json(self.write_json_root, {"msg": "所有图片处理结束"})


def main(kwargs):
    worker = Slice_process(kwargs)
    worker.run()
