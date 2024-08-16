"""
@Description: 
@Author: Adiazhang
@Date: 2024-06-12 11:26:24
@LastEditTime: 2024-06-24 21:43:55
@LastEditors: Adiazhang
"""

"""
@Description: 
@Author: Adiazhang
@Date: 2024-06-12 11:26:24
@LastEditTime: 2024-06-12 16:12:45
@LastEditors: Adiazhang
"""

"""
@Description: 
@Author: Adiazhang
@Date: 2024-06-12 11:26:24
@LastEditTime: 2024-06-12 11:50:23
@LastEditors: Adiazhang
"""
import os
from filelock import FileLock
import time

from PySide2.QtWidgets import (
    QApplication,
    QMessageBox,
    QFileDialog,
    QMainWindow,
)
from PySide2.QtGui import QCloseEvent
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import Signal, QThread, QFile
import pyqtgraph as pg

from utils.Slice_process_utils import main as Slice_main
from utils.EFI_process_utils import main as EFI_main
from utils.draw_utils import Draw_Save
from utils.reconstruction_utils import main as Recon_main


import multiprocessing as mp
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class read_json(QThread):
    text_print = Signal(str)
    stop_signal = Signal()
    infor_print = Signal(str)
    image_draw = Signal(dict)
    bar_ini = Signal(list)
    bar_clear = Signal()
    bar_update = Signal(int)
    return_result = Signal(list)

    def __init__(self, root):
        super().__init__()
        self.read_json_root = root
        self.lock = FileLock(self.read_json_root + ".lock")
        self.bar_value = 0

    def run(self):
        previous_message = None

        while True:
            if os.path.exists(self.read_json_root):
                with self.lock:
                    try:
                        with open(self.read_json_root, "r") as f:

                            message = json.load(f)

                            if message != previous_message:
                                print(message)
                                previous_message = message
                                if "msg" in message:
                                    self.text_print.emit(message["msg"] + "\n")
                                if "bar_clear" in message:
                                    self.bar_value = 0
                                    self.bar_clear.emit()
                                if "bar_ini" in message:
                                    self.bar_ini.emit(message["bar_ini"])
                                    # self.bar_end=message['bar_ini'][1]
                                if "bar_update" in message:
                                    self.bar_value += 1
                                    self.bar_update.emit(self.bar_value)
                                if "draw_infor" in message:
                                    self.image_draw.emit(message["draw_infor"])
                                if "result" in message:
                                    if self.mode == "train":
                                        self.return_result.emit(message["result"])
                                    else:
                                        self.return_result.emit(message["result"])
                                if "finish" in message:
                                    self.infor_print.emit("任务完成")
                                    self.lock.release()

                                    break
                    except:
                        pass
                    self.lock.release()


class GUI(QMainWindow):

    def __init__(self):
        super(GUI, self).__init__()
        # 加载UI文件
        loader = QUiLoader()
        loader.registerCustomWidget(pg.PlotWidget)
        ui_file = QFile("HOLOAI_GUI.ui")
        ui_file.open(QFile.ReadOnly)
        self.ui = loader.load(ui_file, self)
        ui_file.close()
        self.setCentralWidget(self.ui)
        self.ui.setWindowTitle("HOLOAI")
        self.tabchange()
        if self.method_type != "Reconstruction":
            self.modechoice()
            self.print_gui_efi(
                "EFI处理的输入模块中参数均为整型；模型参数配置模块中的特征通道扩大倍率为用英文逗号分隔的整数，和引入模型中输入图片的下采样次数相关，如1,2,3意味着输入图片被下采样两次，下采样时特征通道放大2、3倍，除该参数和学习率外，其他参数均为整型；识别参数配置模块中所有参数均为浮点数类型"
                + "\n"
                + "Slice处理的输入模块中均为整型，模型参数配置模块中输出检测框置信度阈值为浮点数类型，其他参数均为整型；识别参数配置模块中所有参数均为浮点数类型"
                + "\n"
            )

        self.EFI_RES = -1
        self.SLICE_RES = -1
        self.input_draw_config_path = -1
        self.read_result_path = -1
        # self.if_read_result = False
        self.recon_img_path = -1
        self.recon_imgfile_path = -1
        self.bin_interval = int(self.ui.draw_bin.text())
        self.ui.draw.clicked.connect(self.run_draw_result)
        self.ui.save_fig.clicked.connect(self.run_save_fig)
        self.ui.save_res.clicked.connect(self.run_write_result)
        self.ui.print_para.clicked.connect(self.print_lwcmvd)

        self.ui.save_root.clicked.connect(self.open_res_root)
        self.ui.input_particle_root.clicked.connect(self.open_input_res_file)
        self.ui.read_input_particle.clicked.connect(self.read_input_draw)
        self.ui.input_exp_root.clicked.connect(self.open_input_draw_config_file)
        self.ui.read_input_exp.clicked.connect(self.read_input_draw_config)

    def closeEvent(self, event: QCloseEvent):
        try:
            if self.thread and self.thread.isRunning():
                self.thread.terminate()
                if self.thread.isFinished():
                    self.thread.deleteLater()
        except:
            pass
        reply = QMessageBox.question(
            self,
            "本程序",
            "是否要退出程序？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def tabchange(self):
        reply = QMessageBox.question(
            self.ui,
            "模式选择",
            "是否切换模式？默认为重建",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )

        if reply == QMessageBox.Yes:
            self.ui.tabWidget.setTabEnabled(2, False)
            sub_reply = QMessageBox.question(
                self.ui,
                "方法选择",
                "是否切换模式？默认为EFI处理",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if sub_reply == QMessageBox.Yes:
                self.ui.tabWidget.setTabEnabled(0, False)
                self.method_type = "Slice_process"
                self.read_json_root = "logging\logging_slice.json"
                self.ui.read_data_slice.clicked.connect(self.read_data_slice)
                self.ui.process_run_slice.clicked.connect(self.Slice_run)
                self.ui.stop_slice.clicked.connect(self.stop_thread)
                self.ui.display_train_root_slice.clicked.connect(
                    self.open_data_yaml_file
                )
                self.ui.train_model_config_slice.clicked.connect(
                    self.open_modelconfig_yaml_file
                )

                self.ui.display_inferroot_slice.clicked.connect(self.open_filefold)
                self.ui.display_infer_weight_root_slice.clicked.connect(
                    self.open_weight_file
                )

            else:

                self.ui.tabWidget.setTabEnabled(1, False)
                self.method_type = "EFI_process"
                self.read_json_root = "logging\logging_EFI.json"
                self.ui.process_run.clicked.connect(self.EFI_run)
                self.ui.read_data_EFI.clicked.connect(self.read_data_EFI)
                self.ui.stop.clicked.connect(self.stop_thread)
                self.ui.display_train_root.clicked.connect(self.open_filefold)
                self.ui.display_inferroot.clicked.connect(self.open_filefold)
                self.ui.display_infer_weight_root.clicked.connect(self.open_weight_file)

        else:
            self.ui.tabWidget.setTabEnabled(1, False)
            self.ui.tabWidget.setTabEnabled(0, False)
            self.method_type = "Reconstruction"
            self.read_json_root = "logging\logging_recon.json"
            self.ui.read_recon.clicked.connect(self.read_data_Recon)
            self.ui.recon_run.clicked.connect(self.Reconstruct_run)
            self.ui.recon_imgroot.clicked.connect(self.open_img_file)
            self.ui.recon_imgfileroot.clicked.connect(self.open_filefold)
            self.ui.recon_save_root.clicked.connect(self.open_res_root)
            self.ui.recon_stop.clicked.connect(self.stop_thread)
            self.ui.clear_root.clicked.connect(self.clear_root)

    def modechoice(self):
        reply = QMessageBox.question(
            self.ui,
            "模式选择",
            "是否切换为训练模型？默认模型为推理模型",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if reply == QMessageBox.Yes:
            if self.method_type == "EFI_process":
                self.ui.if_train.setChecked(True)
            else:
                self.ui.if_train_slice.setChecked(True)
            self.mode = "train"

        else:
            if self.method_type == "EFI_process":
                self.ui.if_infer.setChecked(True)
            else:
                self.ui.if_infer_slice.setChecked(True)
            self.mode = "infer"

    def read_data_slice(self):
        self.bin_interval = int(self.ui.draw_bin.text())
        self.slice_process_kwargs = {"device": "cuda"}
        if self.ui.buttonGroup_slice.checkedButton().text() == "训练":
            self.mode = "train"
            # self.slice_process_kwargs["task"] = self.mode
            # self.slice_process_kwargs["dataset"] = self.train_root_slice

            # self.train_epoch_slice = int(self.ui.train_epoch_slice.text())
            # self.train_batchsize_slice = int(self.ui.train_batchsize_slice.text())

            # self.slice_process_kwargs["epoch"] = self.train_epoch_slice
            # self.slice_process_kwargs["batchsize"] = self.train_batchsize_slice

            # self.slice_process_kwargs["model_config_path"] = (
            #     self.model_config_slice_path
            # )
            # self.print_gui_efi("Slice_process 训练参数读取成功！")
            try:
                self.slice_process_kwargs["task"] = self.mode
                self.slice_process_kwargs["dataset"] = self.train_root_slice

                self.train_epoch_slice = int(self.ui.train_epoch_slice.text())
                self.train_batchsize_slice = int(self.ui.train_batchsize_slice.text())

                self.slice_process_kwargs["epoch"] = self.train_epoch_slice
                self.slice_process_kwargs["batch_size"] = self.train_batchsize_slice

                self.slice_process_kwargs["model_config_path"] = (
                    self.model_config_slice_path
                )
                self.print_gui_efi("Slice_process 训练参数读取成功！")
            except:
                self.warn_box_gui("警告", "输入参数格式有误，请检查！")
        else:
            self.mode = "infer"
            self.slice_process_kwargs["task"] = self.mode
            # try:

            # 图片原始路径下文件夹结构
            self.slice_process_kwargs["save_run_root"] = self.save_result_path
            self.slice_process_kwargs["reconstructed_img_root"] = self.infer_root_slice
            self.start_infer_id_slice = int(self.ui.infer_start_slice.text())
            self.end_infer_id_slice = int(self.ui.infer_end_slice.text())

            self.slice_process_kwargs["startid"] = self.start_infer_id_slice
            self.slice_process_kwargs["endid"] = self.end_infer_id_slice

            self.infer_H_multiple_slice = int(self.ui.infer_H_multiple_slice.text())
            self.infer_W_multiple_slice = int(self.ui.infer_W_multiple_slice.text())

            self.slice_process_kwargs["H_multiple"] = self.infer_H_multiple_slice
            self.slice_process_kwargs["W_multiple"] = self.infer_W_multiple_slice

            self.infer_raw_resolution_slice = (
                int(self.ui.infer_raw_H_slice.text()),
                int(self.ui.infer_raw_W_slice.text()),
            )
            self.infer_ifcrop_slice = not self.ui.infer_ifcrop_slice.isChecked()
            self.slice_process_kwargs["need_cut"] = self.infer_ifcrop_slice

            if self.infer_ifcrop_slice:
                self.cropped_imagesize_slice = [
                    int(self.ui.infer_crop_H_slice.text()),
                    int(self.ui.infer_crop_W_slice.text()),
                ]
                self.slice_process_kwargs["cropped_imgshape"] = (
                    self.cropped_imagesize_slice
                )
                if self.end_infer_id_slice == -1:
                    self.img_name_list_slice = os.listdir(
                        os.path.join(self.infer_root_slice, "no_crop")
                    )[self.start_infer_id_slice :]
                else:
                    self.img_name_list_slice = os.listdir(
                        os.path.join(self.infer_root_slice, "no_crop")
                    )[self.start_infer_id_slice : self.end_infer_id_slice + 1]

            else:
                self.slice_process_kwargs["cropped_imgshape"] = None
                self.img_name_list = os.listdir(
                    os.path.join(self.infer_root_slice, "no_crop")
                )

            self.slice_process_kwargs["weights"] = self.infer_weight_path_slice

            self.output_detect_thre = float(self.ui.detection_thre.text())
            self.save_detect_img = self.ui.if_save_detectimg.isChecked()

            self.slice_process_kwargs["detection_conf"] = self.output_detect_thre
            self.slice_process_kwargs["save_detection_img"] = self.save_detect_img

            self.maxd_slice = float(self.ui.max_d.text())
            self.mind_slice = float(self.ui.min_d.text())
            self.maxe_slice = float(self.ui.max_e.text())
            self.mine_slice = float(self.ui.min_e.text())
            self.max_eula_slice = float(self.ui.max_eula.text())
            self.min_eula_slice = float(self.ui.min_eula.text())
            self.cal_fused_slice_num_slice = float(
                self.ui.num_fusedslices_seg_slice.text()
            )
            self.merge_iou = float(self.ui.merge_iou.text())

            self.startz_slice = float(self.ui.start_recon_posi_slice.text())
            self.endz_slice = float(self.ui.end_recon_posi_slice.text())

            self.interval_slice = float(self.ui.recon_inter_slice.text())
            self.pixel_slice = float(self.ui.pixelsize_slice.text())
            self.post_process_config_slice = {}
            self.post_process_config_slice["maxd"] = self.maxd_slice
            self.post_process_config_slice["mind"] = self.mind_slice
            self.post_process_config_slice["maxe"] = self.maxe_slice
            self.post_process_config_slice["mine"] = self.mine_slice
            self.post_process_config_slice["max_eula"] = self.max_eula_slice
            self.post_process_config_slice["min_eula"] = self.min_eula_slice
            self.post_process_config_slice["cal_interval"] = (
                self.cal_fused_slice_num_slice
            )
            self.post_process_config_slice["mergeiou"] = self.merge_iou
            self.slice_process_kwargs["pp_config"] = self.post_process_config_slice

            # except:
            #     self.warn_box_gui("警告", "输入参数格式有误，请检查！")
            self.draw_save_config = {
                "minz": self.start_infer_id_slice,
                "particle": None,
                "pixel": self.pixel_slice,
                "deltaz": self.endz_slice - self.startz_slice,
                "W": self.infer_raw_resolution_slice[1],
                "H": self.infer_raw_resolution_slice[0],
                "bin_interval": self.bin_interval,
                "save_path": self.save_result_path,
                "method_type": self.method_type,
                "pp_config": self.post_process_config_slice,
                "img_name_list": self.img_name_list_slice,
            }
            self.print_gui_efi("Slice_process 推理参数读取成功！")

    def read_data_EFI(self):

        try:
            if self.ui.buttonGroup_EFI.checkedButton().text() == "训练":
                self.mode = "train"
                try:
                    self.bin_interval = int(self.ui.draw_bin.text())
                    self.model_config = {}
                    self.train_ifcrop = not self.ui.train_ifcrop.isChecked()
                    if self.train_ifcrop:
                        self.cropped_imagesize = (
                            int(self.ui.train_cropped_H.text()),
                            int(self.ui.train_cropped_W.text()),
                        )

                    self.epoch = int(self.ui.epoch.text())
                    self.train_batchsize = int(self.ui.train_batchsize.text())
                    self.lr = float(self.ui.lr.text())

                    # 英文逗号间隔

                    self.train_raw_resolution = (
                        int(self.ui.train_raw_H.text()),
                        int(self.ui.train_raw_W.text()),
                    )

                    self.train_H_multiple = int(self.ui.train_H_multiple.text())
                    self.train_W_multiple = int(self.ui.train_W_multiple.text())
                    self.train_ifiniprep_efi = self.ui.efi_train_ifiniprep.isChecked()
                    self.save_weight_path = os.path.join(
                        r"GUI_weight_root\EFI_process",
                        time.strftime("%Y_%m-%d %H_%M_%S", time.localtime()),
                    )
                    try:
                        os.mkdir(self.save_weight_path)
                    except:
                        pass
                    self.print_gui_efi("EFI_process训练参数读取成功！" + "\n")
                except:
                    self.warn_box_gui("警告", "输入参数格式有误，请检查！")
            elif self.ui.buttonGroup_EFI.checkedButton().text() == "推理":
                self.mode = "infer"
                # 输入group中参数获取
                # try:
                # self.infer_ifiniprep_efi = self.ui.efi_infer_ifiniprep.isChecked()
                self.bin_interval = int(self.ui.draw_bin.text())
                self.start_infer_id = int(self.ui.infer_start.text())
                self.end_infer_id = int(self.ui.infer_end.text())

                self.infer_ifcrop = not self.ui.infer_ifcrop.isChecked()

                if self.infer_ifcrop:
                    self.cropped_imagesize = (
                        int(self.ui.infer_crop_H.text()),
                        int(self.ui.infer_crop_W.text()),
                    )
                    self.img_name_list = os.listdir(self.infer_root, "crop", "image")
                self.img_name_list = os.listdir(self.infer_root, "uncrop", "image")
                self.infer_raw_resolution = (
                    int(self.ui.infer_raw_H.text()),
                    int(self.ui.infer_raw_W.text()),
                )
                self.infer_H_multiple = int(self.ui.infer_H_multiple.text())
                self.infer_W_multiple = int(self.ui.infer_W_multiple.text())
                # 模型参数group中参数获取

                self.infer_batchsize = int(self.ui.infer_batchsize.text())

                self.maxd = float(self.ui.max_d.text())
                self.mind = float(self.ui.min_d.text())
                self.maxe = float(self.ui.max_e.text())
                self.mine = float(self.ui.min_e.text())
                self.max_eula = float(self.ui.max_eula.text())
                self.min_eula = float(self.ui.min_eula.text())
                self.cal_fused_slice_num = float(self.ui.num_fusedslices_seg.text())
                self.startz = float(self.ui.start_recon_posi.text())
                self.endz = float(self.ui.end_recon_posi.text())
                self.interval = float(self.ui.recon_inter.text())
                self.pixel = float(self.ui.pixelsize.text())

                self.post_process_config = {}
                self.post_process_config["maxd"] = self.maxd
                self.post_process_config["mind"] = self.mind
                self.post_process_config["maxe"] = self.maxe
                self.post_process_config["mine"] = self.mine
                self.post_process_config["max_eula"] = self.max_eula
                self.post_process_config["min_eula"] = self.min_eula

                self.print_gui_efi("EFI_process推理参数读取成功！" + "\n")
                # except:
                #     self.warn_box_gui("警告", "输入参数格式有误，请检查！")
                self.draw_save_config = {
                    "minz": self.startz,
                    "particle": None,
                    "pixel": self.pixel,
                    "deltaz": self.endz - self.startz,
                    "W": self.infer_raw_resolution[1],
                    "H": self.infer_raw_resolution[0],
                    "bin_interval": self.bin_interval,
                    "save_path": self.save_result_path,
                    "method_type": self.method_type,
                    "pp_config": self.post_process_config,
                    "img_name_list": self.img_name_list,
                }

                # self.print_gui_efi("保存数据类创建成功！" + "\n")
        except:
            self.warn_box_gui("警告", "请先选择运行模式！")

    def read_data_Recon(self):
        try:
            self.recon_wave = int(self.ui.recon_wavelength.text()) * 1e-9
            self.recon_zmin = float(self.ui.start_recon.text()) * 1e-3
            self.recon_zmax = float(self.ui.recon_end.text()) * 1e-3
            self.recon_interval = float(self.ui.recon_interval.text()) * 1e-3
            self.recon_pixel = float(self.ui.recon_pixel_size.text()) * 1e-6
            self.Nx = int(self.ui.recon_img_W.text())
            self.Ny = int(self.ui.recon_img_H.text())
            self.ifoffaxis = self.ui.if_offaxis.isChecked()
            self.ifsaveefi = self.ui.if_save_efi.isChecked()
            self.ifsaveslice = self.ui.if_save_slice.isChecked()

            self.print_gui_efi("重建参数读取成功！" + "\n")

        except:
            self.warn_box_gui("警告", "输入参数格式有误，请检查！")

    def generate_draw_save_config(self):
        self.draw_save_config = {}
        self.draw_save_config["particle"] = self.input_res
        self.draw_save_config["H"] = self.input_draw_config["H"]
        self.draw_save_config["W"] = self.input_draw_config["W"]
        self.draw_save_config["pixel"] = self.input_draw_config["pixel"]
        self.draw_save_config["deltaz"] = self.input_draw_config["deltaz"]
        self.draw_save_config["minz"] = None
        self.draw_save_config["bin_interval"] = self.bin_interval
        self.draw_save_config["method_type"] = self.method_type
        self.draw_save_config["save_path"] = self.save_result_path
        self.draw_save_config["pp_config"] = None
        self.draw_save_config["img_name_list"] = [
            i for i in range(self.input_draw_config["num_image"])
        ]

    def return_result_from_thread(self, list):
        self.EFI_RES = list

    def return_result_from_slice(self, list):
        self.SLICE_RES = list

    def clear_root(self):
        self.recon_img_path = -1
        self.recon_imgfile_path = -1
        self.save_recon_root = -1
        self.print_gui_efi("清空路径成功！" + "\n")

    def stop_thread(self):
        # self.stop_event.set()
        # print(self.stop_event.is_set())
        self.process.terminate()
        del self.process
        # self.process.waitForFinished(3000)
        # self.process.kill()  # 如果子进程没有正常结束，强制终止
        # self.process.deleteLater()  # 释放 QProcess 资源
        self.process = None
        self.thread.terminate()
        if self.thread.isFinished():
            self.thread.deleteLater()
        self.print_gui_efi("进程和线程已停止！" + "\n")
        self.infor_box_gui("进程和线程已停止！")

    def warn_box_gui(self, head, warn):
        warn_box = QMessageBox()
        warn_box.warning(self.ui, head, warn)

    def infor_box_gui(self, infor):
        infor_box = QMessageBox()
        infor_box.information(self.ui, "通知", infor)

    def ini_process_bar_gui_efi(self, list):
        self.ui.progressBar_EFI.setRange(list[0], list[1])
        self.ui.progressBar_EFI.setValue(0)

    def clear_process_bar_gui(self):
        self.ui.progressBar_EFI.reset()

    def update_process_bar_gui(self, int):
        self.ui.progressBar_EFI.setValue(int)

    def draw_gui(self, dict):
        # My_Thread_draw = My_Thread_draw(guidrawer, dict)
        # My_Thread_draw.start()
        x = dict["x_data"]
        y1 = dict["y1_data"]
        y2 = dict["y2_data"]
        legend = dict["legend"]
        x_label = dict["x_label"]
        y_label = dict["y_label"]

        self.ui.show_EFI.clear()
        self.ui.show_EFI.setLabel("bottom", x_label)
        self.ui.show_EFI.setLabel("left", y_label)
        self.ui.show_EFI.setTitle("Loss curve for training and validation")
        self.ui.show_EFI.addLegend()
        self.ui.show_EFI.showGrid(x=True, y=True)
        self.ui.show_EFI.plot(x, y1, pen="r", name=legend[0])
        self.ui.show_EFI.plot(x, y2, pen="g", name=legend[1])
        # self.ui.show_EFI.legend.append(legend[0])
        # self.ui.show_EFI.legend.append(legend[1])
        self.ui.show_EFI.show()

    def print_gui_efi(self, text):
        self.ui.output_print.append(text)
        self.ui.output_print.ensureCursorVisible()

    def open_filefold(self):
        # try:
        if self.method_type != "Reconstruction":

            self.print_gui_efi("你正在选择导入 " + self.mode + "文件夹" + "\n")
        filePath = QFileDialog.getExistingDirectory(
            self.ui,  # 父窗口对象
            "选择你要导入的文件夹",  # 标题
        )

        if filePath:

            # self.ui.output_print.append("你选择的文件夹路径为: " + filePath + "\n")
            # self.ui.output_print.ensureCursorVisible()

            self.print_gui_efi("你选择的文件夹路径为: " + filePath + "\n")
            if self.method_type == "EFI_process":
                if self.mode == "train":
                    self.train_root = filePath
                else:
                    self.infer_root = filePath
            elif self.method_type == "Slice_process":
                self.infer_root_slice = filePath
            else:
                self.recon_imgfile_path = filePath
        else:
            self.print_gui_efi("你未选择文件夹路径，请重新选择" + "\n")
            # self.ui.output_print.append("你未选择文件夹路径，请重新选择" + "\n")
            # self.ui.output_print.ensureCursorVisible()
        # except:
        #     self.warn_box_gui("警告", "请先选择运行模式！")

    def open_res_root(self):
        filePath = QFileDialog.getExistingDirectory(
            self.ui,  # 父窗口对象
            "E:\Adia\code\HOLO-main_multiprocess\\",
            "选择你要导入的文件夹",  # 标题
        )
        self.print_gui_efi("你正在选择结果保存文件夹" + "\n")
        if filePath:
            if self.method_type != "Reconstruction":
                self.save_result_path = filePath
            else:
                self.save_recon_root = filePath
            # self.ui.output_print.append("你选择的文件夹路径为: " + filePath + "\n")
            # self.ui.output_print.ensureCursorVisible()
            self.print_gui_efi("你选择的文件夹路径为: " + filePath + "\n")
            self.save_result_path = filePath
        else:
            self.print_gui_efi("你未选择文件夹路径，请重新选择" + "\n")

    def open_input_res_file(self):
        fileName, _ = QFileDialog.getOpenFileName(
            self.ui,
            "选择你要导入的文件",
            "E:\Adia\code\HOLO-main_multiprocess",
            "Result Files (*.csv *.xls *.xlsx *.txt)",
        )
        self.print_gui_efi("你正在选择导入结果文件" + "\n")
        if fileName:
            # self.ui.output_print.append("你选择的文件路径为: " + fileName + "\n")
            self.print_gui_efi("你选择的文件路径为: " + fileName + "\n")
            self.read_result_path = fileName
        else:
            self.print_gui_efi("你未选择文件路径，请重新选择" + "\n")

    def open_weight_file(self):
        fileName, _ = QFileDialog.getOpenFileName(
            self.ui,
            "选择你要导入的文件",
            "E:\Adia\code\HOLO-main_multiprocess\\",
            "Weight Files (*.pt)",
        )
        self.print_gui_efi("你正在选择导入权重文件" + "\n")
        if fileName:
            # self.ui.output_print.append("你选择的文件路径为: " + fileName + "\n")
            self.print_gui_efi("你选择的文件路径为: " + fileName + "\n")
            if self.method_type == "Slice_process":
                self.infer_weight_path_slice = fileName
            else:
                self.weight_path = fileName

        else:
            self.print_gui_efi("你未选择文件路径，请重新选择" + "\n")
            # self.ui.output_print.append("你未选择文件路径，请重新选择" + "\n")

    def open_data_yaml_file(self):
        fileName, _ = QFileDialog.getOpenFileName(
            self.ui,
            "选择你要导入的文件",
            "E:\Adia\code\HOLO-main_multiprocess",
            "Configuration Files (*.yaml)",
        )
        self.print_gui_efi("你正在选择导入模型配置文件" + "\n")
        if fileName:
            self.print_gui_efi("你选择的文件路径为: " + fileName + "\n")
            # 显示yaml文件中的内容

            self.train_root_slice = fileName

        else:
            self.print_gui_efi("你未选择文件路径，请重新选择" + "\n")
            # self.ui.output_print.append("你未选择文件路径，请重新选择" + "\n")

    def open_modelconfig_yaml_file(self):
        fileName, _ = QFileDialog.getOpenFileName(
            self.ui,
            "选择你要导入的文件",
            "E:\Adia\code\HOLO-main_multiprocess",
            "Configuration Files (*.yaml)",
        )
        self.print_gui_efi("你正在选择导入模型配置文件" + "\n")
        if fileName:
            self.print_gui_efi("你选择的文件路径为: " + fileName + "\n")
            # 显示yaml文件中的内容
            self.model_config_slice_path = fileName
        else:
            self.print_gui_efi("你未选择文件路径，请重新选择" + "\n")
            # self.ui.output_print.append("你未选择文件路径，请重新选择" + "\n")

    def open_img_file(self):
        fileName, _ = QFileDialog.getOpenFileName(
            self.ui,
            "选择你要导入的文件",
            "E:\Adia\code\HOLO-main_multiprocess",
            "Image Files (*.jpg *.png *.bmp *.tiff *.tif)",
        )
        if fileName:
            self.print_gui_efi("你选择的文件路径为: " + fileName + "\n")
            # 显示yaml文件中的内容
            self.recon_img_path = fileName
        else:
            self.print_gui_efi("你未选择文件路径，请重新选择" + "\n")

    def monitor_thread(self):

        self.thread = read_json(self.read_json_root)

        # self.print_gui_efi("类实例化完毕!" + "\n")

        self.thread.text_print.connect(self.print_gui_efi)
        self.thread.stop_signal.connect(self.stop_thread)
        self.thread.infor_print.connect(self.infor_box_gui)
        self.thread.image_draw.connect(self.draw_gui)
        self.thread.bar_ini.connect(self.ini_process_bar_gui_efi)
        self.thread.bar_clear.connect(self.clear_process_bar_gui)
        self.thread.bar_update.connect(self.update_process_bar_gui)
        self.thread.return_result.connect(self.return_result_from_thread)
        # self.print_gui_efi("信号函数准备完毕!" + "\n")
        self.thread.start()

    def EFI_run(self):
        try:
            os.remove(self.read_json_root)
        except:
            pass
        self.monitor_thread()

        if self.mode == "train":
            self.thwargs = {
                "mode": self.mode,
                "device": "cuda",
                "raw_traindata_root": self.train_root,
                "raw_inference_data_root": None,
                "batch_size": self.train_batchsize,
                "lr": self.lr,
                "num_epochs": self.epoch,
                "if_ini_pre": self.train_ifiniprep_efi,
                "infere_weight_path": None,
                "save_weight_path": self.save_weight_path,
                "need_img_crop": self.train_ifcrop,
                "W_multiple": self.train_W_multiple,
                "H_multiple": self.train_H_multiple,
                "raw_image_shape": self.train_raw_resolution,
                "post_process_config": None,
            }

        else:
            self.thwargs = {
                "mode": self.mode,
                "device": "cuda",
                "raw_traindata_root": None,
                "raw_inference_data_root": self.infer_root,
                "batch_size": self.infer_batchsize,
                "lr": None,
                "num_epochs": None,
                "if_ini_pre": None,
                "infere_weight_path": self.weight_path,
                "save_weight_path": None,
                "need_img_crop": self.infer_ifcrop,
                "W_multiple": self.infer_W_multiple,
                "H_multiple": self.infer_H_multiple,
                "raw_image_shape": self.infer_raw_resolution,
                "post_process_config": self.post_process_config,
            }

        self.process = mp.Process(target=EFI_main, args=(self.thwargs,))
        self.process.start()

    def Reconstruct_run(self):
        try:
            os.remove(self.read_json_root)
        except:
            pass
        self.monitor_thread()
        self.recon_batch_kwargs = {
            "single_holo": self.recon_img_path,
            "rawholoroot": self.recon_imgfile_path,
            "zmin": self.recon_zmin,
            "zmax": self.recon_zmax,
            "interval": self.recon_interval,
            "lamda": self.recon_wave,
            "dx": self.recon_pixel,
            "device": "cuda:0",
            "ifEFI": self.ifsaveefi,
            "Nx": self.Nx,
            "ifsaveslice": self.ifsaveslice,
            "save_root": self.save_recon_root,
            "type": self.ifoffaxis,
            "Ny": self.Ny,
            # "theta": None,
            # "auto_with_bbox": None,
        }  # try:
        self.process = mp.Process(target=Recon_main, args=(self.recon_batch_kwargs,))
        self.process.start()

    def Slice_run(self):
        try:
            os.remove(self.read_json_root)
        except:
            pass
        self.monitor_thread()
        # try:
        if self.mode == "train":
            self.slice_process_kwargs["save_run_root"] = None
            self.slice_process_kwargs["reconstructed_img_root"] = None

            self.slice_process_kwargs["startid"] = None
            self.slice_process_kwargs["endid"] = None
            self.slice_process_kwargs["H_multiple"] = None
            self.slice_process_kwargs["W_multiple"] = None
            self.slice_process_kwargs["need_cut"] = None
            self.slice_process_kwargs["cropped_imgshape"] = None
            self.slice_process_kwargs["weights"] = None
            self.slice_process_kwargs["detection_conf"] = None
            self.slice_process_kwargs["save_detection_img"] = None
            self.slice_process_kwargs["pp_config"] = None

        else:
            self.slice_process_kwargs["dataset"] = None
            self.slice_process_kwargs["epoch"] = None
            self.slice_process_kwargs["batch_size"] = None
            self.slice_process_kwargs["model_config_path"] = None

        self.process = mp.Process(target=Slice_main, args=(self.slice_process_kwargs,))
        self.process.start()

    #     except:
    #         self.warn_box_gui("警告", "请确认已点击参数读取按钮或参数输入类型正确")

    def read_input_draw(self):
        self.input_res = np.array(pd.read_excel(self.read_result_path)).tolist()
        # print(self.input_res)

        self.if_read_result = True
        self.ui.save_res.setEnabled(False)
        self.print_gui_efi("导入绘图颗粒场文件读取成功" + "\n")

    def open_input_draw_config_file(self):
        fileName, _ = QFileDialog.getOpenFileName(
            self.ui,
            "选择你要导入的文件",
            "E:\Adia\code\HOLO-main_multiprocess",
            "Configuration Files (*.txt)",
        )
        self.print_gui_efi("你正在选择导入绘图参数配置文件" + "\n")
        if fileName:
            self.print_gui_efi("你选择的文件路径为: " + fileName + "\n")
            # 显示yaml文件中的内容
            self.input_draw_config_path = fileName
        else:
            self.print_gui_efi("你未选择文件路径，请重新选择" + "\n")
            # self.ui.output_print.append("你未选择文件路径，请重新选择" + "\n")

    def read_input_draw_config(self):
        self.input_draw_config = {}
        with open(self.input_draw_config_path) as f:
            lines = f.readlines()
        for line in lines:
            key, value = line.strip().split(" ")[0], eval(line.strip().split(" ")[1])
            # self.input_draw_config[key]=value
            self.input_draw_config.update({key: value})
        print(self.input_draw_config)

        self.print_gui_efi("导入绘图参数配置文件读取成功" + "\n")

    def print_lwcmvd(self):
        if self.input_draw_config_path == -1:
            if self.method_type == "Slice_process":
                self.draw_save_config["particle"] = self.SLICE_RES
            else:
                self.draw_save_config["particle"] = self.EFI_RES

            self.Draw_Save = Draw_Save(self.draw_save_config)
            self.Draw_Save.pre_process()

        else:
            self.generate_draw_save_config()
            self.Draw_Save = Draw_Save(self.draw_save_config)
            self.Draw_Save.cal_parameter()
        self.print_gui_efi("LWC:" + str(self.Draw_Save.lwc) + "g/cm3")
        self.print_gui_efi("MVD:" + str(self.Draw_Save.mvd) + "um")

    def run_draw_result(self):
        if self.input_draw_config_path == -1:
            if self.method_type == "Slice_process":
                self.draw_save_config["particle"] = self.SLICE_RES
                self.draw_save_config["bin_interval"] = int(self.ui.draw_bin.text())
            else:
                self.draw_save_config["particle"] = self.EFI_RES
                self.draw_save_config["bin_interval"] = int(self.ui.draw_bin.text())
            self.Draw_Save = Draw_Save(self.draw_save_config)

            self.Draw_Save.pre_process()
        else:
            self.generate_draw_save_config()
            self.draw_save_config["bin_interval"] = int(self.ui.draw_bin.text())
            self.Draw_Save = Draw_Save(self.draw_save_config)
            self.Draw_Save.cal_parameter()
        self.draw_type_id = self.ui.image_type.currentIndex()
        if self.draw_type_id == 0:
            self.warn_box_gui(
                "警告", "全部模式下不支持显示绘图结果，请重新选择需要绘制的图像类型！"
            )
        # elif self.draw_type_id == 1:
        #     particle = np.array(self.Draw_Save.particle)
        #     d = particle[:, 3]
        #     self.ui.show_EFI.clear()
        #     self.ui.show_EFI.hist(d, bins=self.Draw_Save.bins, density=True)
        #     self.ui.show_EFI.setLabel("bottom", "Diameter")
        #     self.ui.show_EFI.setLabel("left", "Frequency")
        #     self.ui.show_EFI.setTitle("Diameter Distribution")
        #     self.ui.show_EFI.show()
        elif self.draw_type_id == 1:
            self.ui.show_EFI.clear()
            bg = pg.BarGraphItem(
                x=self.Draw_Save.binlist,
                height=self.Draw_Save.cul_volume_ratio,
                width=2,
            )
            self.ui.show_EFI.addItem(bg)
            self.ui.show_EFI.setLabel("bottom", "Diameter")
            self.ui.show_EFI.setLabel("left", "Culmative Volume Ratio")
            self.ui.show_EFI.setTitle("Culmative Volume Ratio")
            self.ui.show_EFI.show()

        elif self.draw_type_id == 2:
            self.ui.show_EFI.clear()

            bg = pg.BarGraphItem(
                x=self.Draw_Save.binlist, height=self.Draw_Save.volume_ratio, width=2
            )
            self.ui.show_EFI.addItem(bg)
            self.ui.show_EFI.setLabel("bottom", "Diameter")
            self.ui.show_EFI.setLabel("left", "Volume Ratio")
            self.ui.show_EFI.setTitle("Volume Ratio")
            self.ui.show_EFI.show()

    def run_save_fig(self):
        self.print_gui_efi("默认保存所有图片")
        if self.input_draw_config_path == -1:
            if self.method_type == "Slice_process":
                self.draw_save_config["particle"] = self.SLICE_RES
            else:
                self.draw_save_config["particle"] = self.EFI_RES
            self.draw_save_config["bin_interval"] = int(self.ui.draw_bin.text())
            self.Draw_Save = Draw_Save(self.draw_save_config)
            self.Draw_Save.pre_process()
        else:
            self.generate_draw_save_config()
            self.draw_save_config["bin_interval"] = int(self.ui.draw_bin.text())
            self.Draw_Save = Draw_Save(self.draw_save_config)
            self.Draw_Save.cal_parameter()
        # self.draw_type = self.draw_image_type_dic[self.draw_type_id]
        particle = np.array(self.Draw_Save.particle)
        d = particle[:, 3]

        plt.hist(d, bins=self.Draw_Save.bins, density=True)
        plt.xlabel("Diameter")
        plt.ylabel("Frequency")
        plt.title("Diameter Distribution")
        plt.savefig(os.path.join(self.save_result_path, "Diameter Distribution.png"))

        plt.close()
        plt.bar(self.Draw_Save.binlist, self.Draw_Save.cul_volume_ratio, width=2)
        plt.xlabel("Diameter")
        plt.ylabel("Culmative Volume Ratio")
        plt.title("Culmative Volume Ratio")
        plt.savefig(os.path.join(self.save_result_path, "Culmative Volume Ratio.png"))

        plt.close()
        plt.bar(self.Draw_Save.binlist, self.Draw_Save.volume_ratio, width=2)
        plt.xlabel("Diameter")
        plt.ylabel("Volume Ratio")
        plt.title("Volume Ratio")
        plt.savefig(os.path.join(self.save_result_path, "Volume Ratio.png"))

        plt.close()

        self.print_gui_efi("统计结果图已保存")

    def run_write_result(self):
        if self.method_type == "Slice_process":
            self.draw_save_config["particle"] = self.SLICE_RES
        else:
            self.draw_save_config["particle"] = self.EFI_RES
        self.Draw_Save = Draw_Save(self.draw_save_config)
        self.Draw_Save.pre_process()
        self.Draw_Save.write_result()
        self.print_gui_efi("统计结果已保存(txt,csv)")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    app = QApplication([])
    holoai = GUI()
    holoai.show()
    app.exec_()
