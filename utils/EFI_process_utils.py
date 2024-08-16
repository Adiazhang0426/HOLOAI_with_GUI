"""
@Description: 
@Author: Adiazhang
@Date: 2024-06-04 10:47:04
@LastEditTime: 2024-06-04 15:20:18
@LastEditors: Adiazhang
"""

from filelock import FileLock
import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2
import os
from EFI_model import unet, res_unet
import random
import shutil
from skimage import measure

from PySide2.QtCore import Signal, QObject, QThread, QRunnable

import json, sys, msvcrt


def normalize(a):
    if type(a) == np.ndarray:
        return (a - np.min(a)) / (np.max(a) - np.min(a))
    else:
        return (a - torch.min(a)) / (torch.max(a) - torch.min(a))


def write_json(path, data):
    lock = FileLock(path + ".lock")
    with lock:
        with open(path, "w") as f:
            json.dump(data, f)
        lock.release()


class EFI_process:
    # text_print = Signal(str)
    # stop_signal = Signal()
    # infor_print = Signal(str)
    # image_draw = Signal(dict)
    # bar_ini = Signal(list)
    # bar_clear = Signal()
    # bar_update = Signal(int)
    # return_result = Signal(list)
    """
    file structure:
    raw_traindata_root->
        |--no_crop->images->
            |--train->
            |--val->
        |--no_crop->labels->
                |--train->
                |--val->
        |--crop->images->
            |--train->
            |--val->
        |--crop->labels->
                |--train->
                |--val->
    raw_inference_data_root->
        |--no_crop->
            |--images
            |--pred_seg

        |--crop->
            |--images
            |--pred_seg
    """

    def __init__(
        self,
        kwargs,
    ):
        super().__init__()
        self.mode = kwargs["mode"]
        self.device = kwargs["device"]
        self.raw_traindata_root = kwargs["raw_traindata_root"]
        self.raw_inference_data_root = kwargs["raw_inference_data_root"]
        self.batch_size = kwargs["batch_size"]
        self.lr = kwargs["lr"]
        self.num_epochs = kwargs["num_epochs"]
        self.if_ini_pre = kwargs["if_ini_pre"]
        self.infere_weight_path = kwargs["infere_weight_path"]
        self.save_weight_path = kwargs["save_weight_path"]
        self.need_img_crop = kwargs["need_img_crop"]
        self.W_multiple = kwargs["W_multiple"]
        self.H_multiple = kwargs["H_multiple"]
        self.raw_image_shape = kwargs["raw_image_shape"]

        self.model = res_unet.Unet(1, 2, 3, 32)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.write_json_root = r"logging/logging_EFI.json"
        if self.need_img_crop == False:
            if self.mode == "train":
                self.dataset_root = os.path.join(self.raw_traindata_root, "no_crop")
            else:
                self.infer_dataset_root = os.path.join(
                    self.raw_inference_data_root, "no_crop"
                )
        else:
            if self.mode == "train":
                self.dataset_root = os.path.join(self.raw_traindata_root, "crop")
            else:
                self.infer_dataset_root = os.path.join(
                    self.raw_inference_data_root, "crop"
                )

        # self.save_object = Draw_Save(kwargs=Draw_Save_config)

    def _crop_img(self, img, label, name):
        if self.mode == "train":

            H, W = self.raw_image_shape
            sw, sh = W // self.W_multiple, H // self.H_multiple
            for i in range(self.H_multiple):
                for j in range(self.W_multiple):
                    flag = random.random()
                    cutimg = img[
                        int(i * sh) : int((i + 1) * sh), int(j * sw) : int((j + 1) * sw)
                    ]
                    cutlabel = label[
                        int(i * sh) : int((i + 1) * sh), int(j * sw) : int((j + 1) * sw)
                    ]
                    if flag >= 0.7:
                        cv2.imwrite(
                            os.path.join(
                                self.dataset_root,
                                "image",
                                "train",
                                name + "_{}_{}.jpg".format(i, j),
                            ),
                            cutimg,
                        )
                        cv2.imwrite(
                            os.path.join(
                                self.dataset_root,
                                "label",
                                "train",
                                name + "_{}_{}.jpg".format(i, j),
                            ),
                            cutlabel,
                        )
                    else:
                        cv2.imwrite(
                            os.path.join(
                                self.dataset_root,
                                "image",
                                "val",
                                name + "_{}_{}.jpg".format(i, j),
                            ),
                            cutimg,
                        )
                        cv2.imwrite(
                            os.path.join(
                                self.dataset_root,
                                "label",
                                "val",
                                name + "_{}_{}.jpg".format(i, j),
                            ),
                            cutlabel,
                        )
        else:
            H, W = self.raw_image_shape
            sw, sh = W // self.W_multiple, H // self.H_multiple
            for i in range(self.H_multiple):
                for j in range(self.W_multiple):
                    cutimg = img[
                        int(i * sh) : int((i + 1) * sh), int(j * sw) : int((j + 1) * sw)
                    ]
                    cv2.imwrite(
                        os.path.join(
                            self.raw_inference_data_root,
                            "crop",
                            "image",
                            name + "_{}_{}.jpg".format(i, j),
                        ),
                        cutimg,
                    )

    def ini_preparation(self):
        if self.mode == "train":
            if self.need_img_crop == True:
                try:
                    os.makedirs(
                        os.path.join(self.raw_traindata_root, "crop", "image", "train")
                    )
                    os.makedirs(
                        os.path.join(self.raw_traindata_root, "crop", "image", "val")
                    )
                    os.makedirs(
                        os.path.join(self.raw_traindata_root, "crop", "label", "val")
                    )
                    os.makedirs(
                        os.path.join(self.raw_traindata_root, "crop", "label", "train")
                    )
                except:
                    print("文件夹已存在")
                    pass
                raw_img_list = os.listdir(
                    os.path.join(self.raw_traindata_root, "no_crop", "image")
                )
                raw_label_list = os.listdir(
                    os.path.join(self.raw_traindata_root, "no_crop", "label")
                )
                for ri, rl in zip(raw_img_list, raw_label_list):
                    img = cv2.imread(
                        os.path.join(self.raw_traindata_root, "no_crop", "image", ri), 0
                    )
                    label = cv2.imread(
                        os.path.join(self.raw_traindata_root, "no_crop", "label", rl), 0
                    )
                    self._crop_img(img, label, ri[:-4])
            else:
                try:
                    os.makedirs(
                        os.path.join(
                            self.raw_traindata_root, "no_crop", "image", "train"
                        )
                    )
                    os.makedirs(
                        os.path.join(self.raw_traindata_root, "no_crop", "image", "val")
                    )
                    os.makedirs(
                        os.path.join(self.raw_traindata_root, "no_crop", "label", "val")
                    )
                    os.makedirs(
                        os.path.join(
                            self.raw_traindata_root, "no_crop", "label", "train"
                        )
                    )
                except:
                    print("文件夹已存在")
                    pass
                raw_img_list = os.listdir(
                    os.path.join(self.raw_traindata_root, "no_crop", "image")
                )
                raw_label_list = os.listdir(
                    os.path.join(self.raw_traindata_root, "no_crop", "label")
                )
                for ri, rl in zip(raw_img_list, raw_label_list):
                    if "." in ri:
                        flag = random.random()
                        if flag < 0.3:
                            shutil.copy(
                                os.path.join(
                                    self.raw_traindata_root, "no_crop", "image", ri
                                ),
                                os.path.join(
                                    self.raw_traindata_root,
                                    "no_crop",
                                    "image",
                                    "train",
                                    ri,
                                ),
                            )
                            shutil.copy(
                                os.path.join(
                                    self.raw_traindata_root, "no_crop", "label", rl
                                ),
                                os.path.join(
                                    self.raw_traindata_root,
                                    "no_crop",
                                    "label",
                                    "train",
                                    rl,
                                ),
                            )
                        else:
                            shutil.copy(
                                os.path.join(
                                    self.raw_traindata_root, "no_crop", "image", ri
                                ),
                                os.path.join(
                                    self.raw_traindata_root,
                                    "no_crop",
                                    "image",
                                    "val",
                                    ri,
                                ),
                            )
                            shutil.copy(
                                os.path.join(
                                    self.raw_traindata_root, "no_crop", "label", rl
                                ),
                                os.path.join(
                                    self.raw_traindata_root,
                                    "no_crop",
                                    "label",
                                    "val",
                                    rl,
                                ),
                            )
        else:
            if self.need_img_crop == True:
                infere_img_list = os.listdir(
                    os.path.join(self.raw_inference_data_root, "no_crop", "image")
                )
                for ii in infere_img_list:
                    img = cv2.imread(
                        os.path.join(
                            self.raw_inference_data_root, "no_crop", "image", ii
                        ),
                        0,
                    )
                    try:
                        os.makedirs(
                            os.path.join(self.raw_inference_data_root, "crop", "image")
                        )
                    except:
                        pass
                    self._crop_img(img, None, ii[:-4])

    def _create_dataset(self):

        if self.mode == "train":
            img_path = os.listdir(os.path.join(self.dataset_root, "image", "train"))
            label_path = os.listdir(os.path.join(self.dataset_root, "label", "train"))
            dataset = list(zip(img_path, label_path))
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            val_img_path = os.listdir(os.path.join(self.dataset_root, "image", "val"))
            val_label_path = os.listdir(os.path.join(self.dataset_root, "label", "val"))
            val_dataset = list(zip(val_img_path, val_label_path))
            val_dataloader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=True
            )
            return dataloader, val_dataloader
        else:
            dataset = os.listdir(os.path.join(self.infer_dataset_root, "image"))
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

            return dataloader

    def read_image(self, img_path, label_path, kind):
        if self.mode == "train":
            img_list = []
            label_list = []
            if kind == "train":
                for i, l in zip(img_path, label_path):
                    img = normalize(
                        cv2.imread(
                            os.path.join(self.dataset_root, "image", "train", i), 0
                        )
                    )
                    label = cv2.imread(
                        os.path.join(self.dataset_root, "label", "train", l), 0
                    )
                    label[label < 125] = 0
                    label[label >= 125] = 1

                    img = np.expand_dims(img, axis=0)
                    # label = np.expand_dims(label, axis=0)
                    img_list.append(torch.tensor(img, dtype=torch.float32))
                    label_list.append(torch.tensor(label, dtype=torch.long))

            else:
                for i, l in zip(img_path, label_path):
                    img = normalize(
                        cv2.imread(
                            os.path.join(self.dataset_root, "image", "val", i), 0
                        )
                    )
                    label = cv2.imread(
                        os.path.join(self.dataset_root, "label", "val", l), 0
                    )
                    label[label < 125] = 0
                    label[label >= 125] = 1
                    img = np.expand_dims(img, axis=0)
                    # label = np.expand_dims(label, axis=0)
                    img_list.append(torch.tensor(img, dtype=torch.float32))
                    label_list.append(torch.tensor(label, dtype=torch.long))
            img_list = torch.stack(img_list, dim=0)
            label_list = torch.stack(label_list, dim=0)
            img_list = img_list.to(self.device)
            label_list = label_list.to(self.device)
            return img_list, label_list

        else:
            img_list = []
            for i in img_path:
                img = normalize(
                    cv2.imread(os.path.join(self.infer_dataset_root, "image", i), 0)
                )
                img_list.append(
                    torch.tensor(np.expand_dims(img, axis=0), dtype=torch.float32)
                )
            img_list = torch.stack(img_list, dim=0)
            img_list = img_list.to(self.device)
            return img_list

    def _train(self):

        # self.text_print.emit(
        #     "------------------------Start training------------------------" + "\n"
        # )
        # print(
        #     json.dumps(
        #         {
        #             "msg": "Start training",
        #             "bar_clear": True,
        #             "bar_ini": [0, self.num_epochs - 1]
        #         }
        #     )
        # )
        write_json(
            self.write_json_root,
            {
                "msg": "Start training",
                "bar_clear": True,
                "bar_ini": [0, self.num_epochs - 1],
            },
        )
        # self.bar_clear.emit()
        # self.bar_ini.emit([0, self.num_epochs - 1])

        train_dataloader, val_data_loader = self._create_dataset()
        ini_loss = 100
        self.model = self.model.to(self.device)
        self.draw_infor = {
            "x_label": "Epoch",
            "y_label": "Loss",
            "legend": ["Train loss", "Val loss"],
            "x_data": [],
            "y1_data": [],
            "y2_data": [],
        }
        all_train_loss = []
        all_val_loss = []
        for epoch in range(self.num_epochs):
            temp_train_loss = []
            temp_val_loss = []
            # if self.stop_event.is_set():
            #     self.stop_signal.emit()
            #     break
            for batch_idx, (img_path, label_path) in enumerate(train_dataloader):

                self.model.train()
                train_img, train_label = self.read_image(img_path, label_path, "train")

                # self.text_print.emit(
                #     "------------------------read train data finished for batch_idx {} in epoch {}------------------------".format(
                #         batch_idx, epoch
                #     )
                #     + "\n"
                # )

                pred_seg = self.model(train_img)
                loss = F.cross_entropy(pred_seg, train_label)
                temp_train_loss.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # print(
                #     json.dumps(
                #         {
                #             "msg": "Backward finished for batch_idx {} in epoch {}".format(
                #                 batch_idx, epoch
                #             )
                #         }
                #     )
                # )
                write_json(
                    self.write_json_root,
                    {
                        "msg": "Backward finished for batch_idx {} in epoch {}".format(
                            batch_idx, epoch
                        )
                    },
                )
                # self.text_print.emit(

                # self.text_print.emit(
                #     "------------------------Backward finished for batch_idx {} in epoch {}------------------------".format(
                #         batch_idx, epoch
                #     )
                #     + "\n",
                # )

                if batch_idx % 20 == 0:
                    # print(
                    #     json.dumps(
                    #         {
                    #             "msg": "Evaluate for batch_idx {} in epoch {}".format(
                    #                 batch_idx, epoch
                    #             )
                    #         }
                    #     )
                    # )
                    write_json(
                        self.write_json_root,
                        {
                            "msg": "Evaluate for batch_idx {} in epoch {}".format(
                                batch_idx, epoch
                            )
                        },
                    )
                    # self.text_print.emit(
                    #     "------------------------Evaluate for batch_idx {} in epoch {}------------------------".format(
                    #         batch_idx, epoch
                    #     )
                    #     + "\n"
                    # )

                    self.model.eval()
                    t_temp_val_loss = []
                    for val_batch_idx, (val_img_path, val_label_path) in enumerate(
                        val_data_loader
                    ):
                        val_img, val_label = self.read_image(
                            val_img_path, val_label_path, "val"
                        )

                        # self.text_print.emit(
                        #     "------------------------read val data finished------------------------"
                        #     + "\n"
                        # )

                        val_pred_seg = self.model(val_img)
                        val_loss = F.cross_entropy(val_pred_seg, val_label)
                        temp_val_loss.append(val_loss.item())
                        t_temp_val_loss.append(val_loss.item())

                    if sum(t_temp_val_loss) / len(t_temp_val_loss) < ini_loss:
                        ini_loss = sum(t_temp_val_loss) / len(t_temp_val_loss)
                        torch.save(
                            self.model.state_dict(),
                            os.path.join(self.save_weight_path, "{}.pt".format(epoch)),
                        )
                        # print(
                        #     json.dumps({"msg": "Model saved".format(batch_idx, epoch)})
                        # )
                        write_json(
                            self.write_json_root,
                            {"msg": "Model saved"},
                        )
                        # self.text_print.emit(
                        #     "------------------------Model saved--------------------------"
                        #     + "\n"
                        # )
                        # self.text_print.emit(
                        #     f"Epoch [{epoch}/{self.num_epochs}], Step [{batch_idx}/{len(train_dataloader)}], Loss: {val_loss.item():.4f}"
                        #     + "\n",
                        # )

            # print(json.dumps({"msg": "Epoch{} finished".format(epoch)}))
            write_json(self.write_json_root, {"msg": "Epoch{} finished".format(epoch)})
            # self.text_print.emit(
            #     "----------------------Epoch{} finished----------------------".format(
            #         epoch
            #     )
            #     + "\n",
            # )

            average_train_loss = sum(temp_train_loss) / len(temp_train_loss)
            average_val_loss = sum(temp_val_loss) / len(temp_val_loss)
            all_train_loss.append(average_train_loss)
            all_val_loss.append(average_val_loss)
            # print(
            #     json.dumps(
            #         {
            #             "msg": f"Epoch [{epoch}/{self.num_epochs}], Train_loss: {average_train_loss:.4f}, Val_loss: {average_val_loss:.4f}"
            #         }
            #     )
            # )
            write_json(
                self.write_json_root,
                {
                    "msg": f"Epoch [{epoch}/{self.num_epochs}], Train_loss: {average_train_loss:.4f}, Val_loss: {average_val_loss:.4f}"
                },
            )
            # self.text_print.emit(
            #     f"Epoch [{epoch}/{self.num_epochs}], Train_loss: {average_train_loss:.4f}, Val_loss: {average_val_loss:.4f}"
            #     + "\n",
            # )
            self.draw_infor["x_data"].append(epoch)
            self.draw_infor["y1_data"].append(average_train_loss)
            self.draw_infor["y2_data"].append(average_val_loss)
            # print(json.dumps({"bar_update": epoch}))
            write_json(self.write_json_root, {"bar_update": str(epoch)})
            # self.bar_update.emit(epoch)

            if epoch // 5 == 0:
                # print(json.dumps({"draw_infor": self.draw_infor}))
                # self.text_print.emit(json.dumps({
                write_json(self.write_json_root, {"draw_infor": self.draw_infor})
        with open(os.path.join(self.save_weight_path, "train_logging.txt"), "w") as f:
            # for key, value in self.model_config:
            #     f.write(key + ":" + str(value) + "\n")
            for i in range(len(all_train_loss)):
                f.write(
                    "epoch: "
                    + str(i)
                    + " train_loss: "
                    + str(all_train_loss[i])
                    + " val_loss: "
                    + str(all_val_loss[i])
                    + "\n"
                )

    def _inference(self):
        write_json(self.write_json_root, {"msg": "start inference"})
        # print(json.dumps({"msg": "start inference"}))
        # self.text_print.emit(
        #     "------------------------start inference------------------------" + "\n",
        # )

        self.model.load_state_dict(torch.load(self.infere_weight_path))
        self.model = self.model.to(self.device)
        write_json(self.write_json_root, {"msg": "load weight finished"})
        # print(json.dumps({"msg": "load weight finished"}))
        # self.text_print.emit(
        #     "------------------------load weight finished------------------------"
        #     + "\n",
        # )

        try:
            self.infer_result_path = os.path.join(self.infer_dataset_root, "pred_seg")
            os.makedirs(self.infer_result_path)
        except:
            pass

        self.model.eval()
        infer_dataloader = self._create_dataset()
        write_json(self.write_json_root, {"msg": "create dataset finished"})
        # print(json.dumps({"msg": "create dataset finished"}))
        write_json(
            self.write_json_root,
            {
                "bar_clear": True,
                "bar_ini": [
                    0,
                    len(os.listdir(os.path.join(self.infer_dataset_root, "image"))),
                ],
            },
        )
        # print(json.dumps({"msg": "start inference"}))
        # print(
        #     json.dumps(
        #         {
        #             "bar_clear": True,
        #             "bar_ini": [
        #                 0,
        #                 len(os.listdir(os.path.join(self.infer_dataset_root, "image"))),
        #             ],
        #         }
        #     )
        # )
        # self.bar_clear.emit()
        # self.bar_ini.emit(
        #     [0, len(os.listdir(os.path.join(self.infer_dataset_root, "image")))]
        # )

        barvalue = 0
        for batch_idx, img_path in enumerate(infer_dataloader):

            infer_img = self.read_image(img_path, None)
            write_json(self.write_json_root, {"msg": "read infer data finished"})
            # print(json.dumps({"msg": "read infer data finished"}))
            # print(json.dumps({"msg": "read infer data finished"}))
            # self.text_print.emit(
            #     "------------------------read infer data finished------------------------"
            #     + "\n",
            # )

            pred_seg = self.model(infer_img)
            pred_seg = torch.argmax(pred_seg, dim=1)
            pred_seg = pred_seg.cpu().numpy()
            pred_seg = pred_seg.astype(np.uint8)
            for i in range(pred_seg.shape[0]):
                seg_name = img_path[i][:-4]
                cv2.imwrite(
                    os.path.join(self.infer_result_path, f"{seg_name}.png"),
                    pred_seg[i],
                )
            # self.text_print.emit(
            #     "------------------------save No.{} infer results finished------------------------"
            #     + "\n",
            # )
            barvalue += 1
            # print(json.dumps({"bar_update": barvalue * self.batch_size}))
            write_json(
                self.write_json_root, {"bar_update": str(barvalue * self.batch_size)}
            )
            # self.bar_update.emit(barvalue * self.batch_size)
            # self.bar_update.emit(barvalue * self.batch_size)

        # if not self.stop_event.is_set():
        #     self.text_print.emit(
        #         "------------------------infer finished------------------------" + "\n",
        #     )
        # print(json.dumps({"msg": "infer finished"}))
        write_json(self.write_json_root, {"msg": "infer finished"})
        # self.text_print.emit(

    def connective_information_extraction(self):
        segimglist = os.listdir(self.infer_result_path)
        particle_information = []
        particle_bbox = []
        min_d, max_d, min_eccentricity, max_eccentricity, min_euler, max_euler = (
            self.pp_config.mind,
            self.pp_config.maxd,
            self.pp_config.mineccetricity,
            self.pp_config.maxeccetricity,
            self.pp_config.mineuler,
            self.pp_config.maxeuler,
        )
        # print(
        #     json.dumps(
        #         {
        #             "msg": "connective information extraction start",
        #             "bar_clear": True,
        #             "bar_ini": [0, len(segimglist)],
        #         }
        #     )
        # )
        write_json(
            self.write_json_root,
            {
                "msg": "connective information extraction start",
                "bar_clear": True,
                "bar_ini": [0, len(segimglist)],
            },
        )
        barvalue = 0
        for segimg in segimglist:
            write_json(self.write_json_root, {"bar_pause": True})
            s_particle_information = []
            s_particle_bbox = []
            segimgpath = os.path.join(self.infer_result_path, segimg)
            segimg = cv2.imread(segimgpath, 0)
            labels = measure.label(segimg, connectivity=2)
            properties = measure.regionprops(labels)
            for prop in properties:
                equal_diameter = prop.equivalent_diameter
                eccentricity = prop.eccentricity
                euler_number = prop.euler_number
                bbox = prop.bbox
                if (
                    equal_diameter >= min_d
                    and equal_diameter <= max_d
                    and eccentricity >= min_eccentricity
                    and eccentricity <= max_eccentricity
                    and euler_number >= min_euler
                    and euler_number <= max_euler
                ):
                    s_particle_information.append(
                        [segimgpath, prop.centroid[1], prop.centroid[0], equal_diameter]
                    )
                    s_particle_bbox.append(
                        [int(bbox[1]), int(bbox[0]), int(bbox[3]), int(bbox[2])]
                    )
            barvalue += 1
            write_json(self.write_json_root, {"bar_update": str(barvalue)})
            # print(json.dumps({"bar_update": barvalue}))
        particle_information.append(s_particle_information)
        particle_bbox.append(s_particle_bbox)
        return particle_information, particle_bbox

    def EFI_z_location(self, particle_bbox):
        particle_z = []
        reconstructed_slice_path = os.path.join(self.raw_inference_data_root, "slices")
        reconstructed_slice_list = os.listdir(reconstructed_slice_path)
        write_json(
            self.write_json_root,
            {
                "msg": "z location start",
                "bar_clear": True,
                "bar_ini": [0, len(particle_bbox)],
            },
        )
        # print(
        #     json.dumps(
        #         {
        #             "msg": "z location start",
        #             "bar_clear": True,
        #             "bar_ini": [0, len(particle_bbox)],
        #         }
        #     )
        # )
        barvalue = 0
        for s_particle_bbox, reconstructed_slice in zip(
            particle_bbox, reconstructed_slice_list
        ):
            write_json(self.write_json_root, {"bar_pause": True})
            s_particle_z = []
            reconstructed_volume = []
            for slice in os.listdir(
                os.path.join(reconstructed_slice_path, reconstructed_slice)
            ):
                slice_img = cv2.imread(
                    os.path.join(reconstructed_slice_path, reconstructed_slice, slice),
                    0,
                )
                reconstructed_volume.append(slice_img)
            reconstructed_volume = np.stack(reconstructed_volume, axis=0)
            for bbox in s_particle_bbox:
                resized_slices = reconstructed_volume[
                    :, bbox[1] : bbox[3], bbox[0] : bbox[2]
                ]
                gv = []
                for i in range(resized_slices.shape[0]):
                    resized_slice = resized_slices[i, ...]
                    sobelx = cv2.Sobel(resized_slice, cv2.CV_64F, 1, 0)
                    sobely = cv2.Sobel(resized_slice, cv2.CV_64F, 0, 1)
                    sobelx = cv2.convertScaleAbs(sobelx)
                    sobely = cv2.convertScaleAbs(sobely)
                    sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
                    gv.append(np.var(sobelxy))
                zid = gv.index(max(gv))
                s_particle_z.append(zid)
            barvalue += 1
            write_json(self.write_json_root, {"bar_update": str(barvalue)})

            particle_z.append(s_particle_z)
        return particle_z

    def run(self):
        all_particle_information = []
        if self.if_ini_pre:
            self.ini_preparation()
            self._create_dataset()

        if self.mode == "train":
            self._train()
            write_json(self.write_json_root, {"finish": True})
            # print(json.dumps({"finish": True}))
        else:
            self._inference()
            particle_information, particle_bbox = (
                self.connective_information_extraction()
            )
            particle_z = self.EFI_z_location(particle_bbox)
            # 恢复成原图大小
            raw_crop_imglist = os.listdir(self.infer_dataset_root)
            # print(json.dumps({"bar_clear":True,"bar_ini":[0,len(raw_crop_imglist)]}))

            # self.bar_clear.emit()
            # self.bar_ini.emit([0, len(raw_crop_imglist)])
            # self.bar_update.emit(0)
            for (
                xyds,
                zs,
                imagename,
            ) in zip(particle_information, particle_z, raw_crop_imglist):

                s_particle_information = []
                for xyd, z in zip(xyds, zs):
                    hid, wid = imagename.split("_")[1], imagename.split("_")[2]
                    xyd[0], xyd[1] = (
                        int(xyd[0])
                        + int(wid) * self.raw_image_shape[1] // self.W_multiple,
                        int(xyd[1])
                        + int(hid) * self.raw_image_shape[0] // self.H_multiple,
                    )
                    s_particle_information.append([int(xyd[0]), int(xyd[1]), z, xyd[2]])

                all_particle_information.append(s_particle_information)

            # self.save_object.particle = all_particle_information
            write_json(self.write_json_root, {"particle": all_particle_information})
            # print(json.dumps({"result": all_particle_information}))
            # self.bar_update.emit(len(raw_crop_imglist))
            # self.save_object.pre_process()
            # self.save_object.write_result()
            write_json(self.write_json_root, {"finish": True})
            # print(json.dumps({"finish": True}))


def main(kwargs):

    EFI_worker = EFI_process(kwargs)

    EFI_worker.run()


if __name__ == "__main__":
    main()
