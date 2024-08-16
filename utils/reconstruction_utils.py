"""
@Description: 
@Author: Adiazhang
@Date: 2024-06-24 14:40:18
@LastEditTime: 2024-06-24 16:03:32
@LastEditors: Adiazhang
"""

import json
from filelock import FileLock
import torch
import numpy as np
import cv2, os
from PySide2.QtCore import QThread, QRunnable
from multiprocessing import Queue, Pool
from functools import partial
import multiprocessing as mp

# 1、勾选离轴角相同或者是频域框相同——>读入背景图——>fft获取频谱图——>自动框选物光的频域分量——>判断哪个物光点是正确的——>再进行批处理
# 2、加相应的GUI上的按键


def write_json(path, data):
    lock = FileLock(path + ".lock")
    with lock:
        with open(path, "w") as f:
            json.dump(data, f)
        lock.release()


def normalize(a):
    if type(a) == np.ndarray:
        return (a - np.min(a)) / (np.max(a) - np.min(a))
    else:
        return (a - torch.min(a)) / (torch.max(a) - torch.min(a))


def torchexp(G_ping: torch.Tensor):
    """
    torch cuda 的复数exp函数
    """
    x = torch.exp(torch.real(G_ping))
    ya = torch.cos(torch.imag(G_ping))
    yb = torch.sin(torch.imag(G_ping))
    G = torch.complex(ya, yb)
    G = x * G
    return G


def fft(img):
    fshift = torch.fft.fftn(img)
    fshift = torch.fft.fftshift(fshift)
    return fshift


def ifft(fshift):
    ifshift = torch.fft.ifftshift(fshift)
    ifshift = torch.fft.ifftn(ifshift)
    return ifshift


# 加单张重建的thread类，写批处理的pooling


class Reconstruction_thread:
    def __init__(
        self,
        kwargs,
    ):
        super().__init__()
        self.kwargs = kwargs

        self.lamda = self.kwargs["lamda"]
        self.dx = self.kwargs["dx"]
        self.Nx = self.kwargs["Nx"]
        self.Ny = self.kwargs["Ny"]
        self.device = self.kwargs["device"]
        self.rawholo = self.kwargs["rawholo"]
        self.zmin = self.kwargs["zmin"]
        self.zmax = self.kwargs["zmax"]
        self.interval = self.kwargs["interval"]
        self.type = self.kwargs["type"]
        self.ifEFI = self.kwargs["ifEFI"]
        self.ifsaveslice = self.kwargs["ifsaveslice"]
        self.save_root = self.kwargs["save_root"]
        self.theta = self.kwargs["theta"]
        self.fftbox = self.kwargs["auto_with_bbox"]
        xx = np.arange(0, self.Nx)
        yy = np.arange(0, self.Ny)
        xx, yy = np.meshgrid(xx, yy)
        self.tempxx = xx
        self.tempyy = yy
        self.k = 2 * np.pi / self.lamda
        self.write_json_root = r"logging\logging_recon.json"

    def generate_kernel_of_propagrate(self):
        dx = self.dx

        fft_fs_x = 1 / dx
        fft_fs_y = 1 / dx
        fft_df_x = fft_fs_x / self.Nx
        fft_df_y = fft_fs_y / self.Ny

        # 原来版本！！！及其重要 有这个才不会出现闪烁
        fx = np.linspace(-fft_fs_x / 2, fft_fs_x / 2, self.Nx, True)
        fy = np.linspace(-fft_fs_y / 2, fft_fs_y / 2, self.Ny, True)
        fx, fy = np.meshgrid(fx, fy)  # 角谱

        fx = torch.from_numpy(fx).to(self.device)
        fy = torch.from_numpy(fy).to(self.device)
        # 到这里为止

        # fx = torch.linspace(-fft_fs_x/2, fft_fs_x/2, Nx)
        # fy = torch.linspace(-fft_fs_y/2, fft_fs_y/2, Ny)
        # fx, fy = torch.meshgrid(fx, fy)  # 角谱

        # fx=fx.to(self.device).T
        # fy=fy.to(self.device).T

        xx = fx * self.lamda
        yy = fy * self.lamda

        kernelTemp = 1j * self.k * torch.sqrt(1 - xx**2 - yy**2)
        return kernelTemp

    def cal_offline_angle(self, theta=None, vis=False, auto_with_bbox=None, **kwargs):
        """计算离轴角

        kwargs
        :param theta_img : str、torch.Tensor 从图片获取离轴角，默认使用实例里面传进来的图像
        :param theta     : 直接输入离轴角 (np.ndarray,np.ndarray)
        :param auto_with_bbox  : 输入一个选择框 (bbox[x,y,w,h])
        """
        Nx = self.Nx
        Ny = self.Ny
        dx = self.dx
        lamda = self.lamda
        if "theta_img" in kwargs:
            img = kwargs["theta_img"]
            if type(img) == str:
                img = self.imread(img)
                img = torch.from_numpy(img)
            if type(img) != torch.Tensor:
                print(" 请输入求离轴角的路径或者图片")
        else:
            img = self.rawholo

        if theta is not None:
            self.theta = theta
            thida1, thida2 = self.theta
            fxc = np.round(np.sin(-thida1) / lamda * dx * Nx + Nx / 2)
            fyc = np.round(np.sin(-thida2) / lamda * dx * Ny + Ny / 2)
        if auto_with_bbox is not None:
            x, y, w, h = auto_with_bbox

        else:
            gui = fft(img).cpu().detach().numpy()
            gui = np.abs(gui)
            gui = (gui / np.max(gui)) * 255

            # if platform.system() == "Windows":
            #     Wscreen, Hscreen = get_real_resolution()
            # else:
            #     raise SystemError(" 请使用Windows系统")
            if auto_with_bbox is not None:
                x, y, w, h = auto_with_bbox
                fftbox = auto_with_bbox

            else:
                Wscreen, Hscreen = 2048, 2048

                ratio = min(Wscreen / gui.shape[1], Hscreen / gui.shape[0]) * 0.8
                # ratio=0.8
                # print(ratio)
                gui_show = cv2.resize(
                    gui, (int(img.shape[1] * ratio), int(img.shape[0] * ratio))
                )
                # gui_show= cv2.resize(gui,(gui.shape[1]//8,gui.shape[0]//8))
                roi = cv2.selectROI(
                    windowName="roi", img=gui_show, showCrosshair=True, fromCenter=False
                )
                cv2.destroyAllWindows()

                x, y, w, h = roi  # 不同尺寸图像

                x, y, w, h = (
                    int(x / ratio),
                    int(y / ratio),
                    int(w / ratio),
                    int(h / ratio),
                )

                fftbox = [x, y, w, h]

            mask = np.zeros((Ny, Nx))
            mask[y : y + int(2 * (h // 2)), x : x + int(2 * (w // 2))] = 1
            gui[mask != 1] = 0

            fyc, fxc = np.where(gui == np.max(gui))

            # thida1 = -np.arcsin((fxc - Nx//2) / Nx / dx * lamda)  # *180/np.pi
            # thida2 = -np.arcsin((fyc - Ny//2) / Ny / dx * lamda)  # *180/np.pi

            thida1 = -np.arcsin(
                (np.round(fxc - Nx / 2 - 1)) / Nx / dx * lamda
            )  # *180/np.pi

            thida2 = -np.arcsin(
                (np.round(fyc - Ny / 2 - 1)) / Ny / dx * lamda
            )  # *180/np.pi
            self.theta = (thida1, thida2)
            self.fftbox = fftbox

            # print(self.theta)
            # self.w=w
            # self.h=h

        # w = max(1 * np.abs((fxc - Nx // 2)), 300)
        # h = max(1 * np.abs((fyc - Ny // 2)), 300)

        # w=1 * np.abs((fxc - Nx // 2))
        # h=1 * np.abs((fyc - Ny // 2))
        a = 0.6 * w
        b = 0.6 * h
        n = 8
        # print(a, b, n)
        self.__generate_refbeam()
        self.__generate_filter(a, b, n)

        self.__generate_fftimg_for_reconstruct()

        return self.theta, self.fftbox

    def __generate_refbeam(self):
        # 生成参考光
        xx = self.tempxx
        yy = self.tempyy
        self.refbeam = np.exp(
            1j * self.k * (xx - self.Nx / 2) * np.sin(self.theta[0]) * self.dx
            + 1j * self.k * (yy - self.Ny / 2) * self.dx * np.sin(self.theta[1])
        )
        self.refbeam = torch.from_numpy(self.refbeam)

    def __generate_filter(self, a, b, n):
        # 生成滤波窗
        xx = self.tempxx
        yy = self.tempyy

        self.filter = np.exp(
            -(((xx - self.Nx / 2) ** 2 / a**2) ** n)
            - ((yy - self.Ny / 2) ** 2 / b**2) ** n
        )
        self.filter = torch.from_numpy(self.filter)

    def __generate_fftimg_for_reconstruct(self, cuda=True):
        # 原图fft准备重建
        img = self.rawholo
        filter = self.filter
        refbeam = self.refbeam
        if cuda == True:
            img = img.cuda()
            filter = filter.cuda()
            refbeam = refbeam.cuda()

        img = img * refbeam
        fshift = fft(img)
        fshift = fshift * filter
        self.offaxis_fftimg = fshift

    def reconstruction_angular_spectrum(self, z):
        """
        返回当前z位置的图像0-1
        """
        kernel = self.generate_kernel_of_propagrate().to(self.device)
        z_real = z
        G_ping = kernel * z_real
        G = torchexp(G_ping)
        if self.type == "inline":
            ifshift = fft(self.rawholo) * G  # 卷积
            ifshift = ifft(ifshift)
        else:
            self.cal_offline_angle(theta=self.theta, auto_with_bbox=self.fftbox)
            ifshift = self.offaxis_fftimg * G  # 卷积
            ifshift = ifft(ifshift)
        img_real = torch.sqrt(torch.real(ifshift) ** 2 + torch.imag(ifshift) ** 2)
        # img_real=normalize(img_real)
        return img_real

    def run(self):
        print("start")
        write_json(self.write_json_root, {"bar_pause": True})
        num_slice = (self.zmax - self.zmin) / self.interval
        if self.type == "offaxis" and self.theta == False and self.fftbox == False:
            self.cal_offline_angle()

        if self.ifEFI:
            efi = torch.ones_like(self.rawholo) * 255

            for i in range(int(num_slice)):

                z = self.zmin + i * self.interval
                recon_slice = normalize(self.reconstruction_angular_spectrum(z)) * 255
                efi = torch.minimum(efi, recon_slice)
                if self.ifsaveslice:
                    recon_slice_cpu = recon_slice.cpu().detach().numpy()
                    cv2.imwrite(
                        os.path.join(
                            self.save_root,
                            "recon_slice_{:0>4d}.png".format(i),
                        ),
                        recon_slice_cpu,
                    )

                del recon_slice
                torch.cuda.empty_cache()

            efi_cpu = efi.cpu().detach().numpy()
            cv2.imwrite(
                os.path.join(self.save_root, "recon_efi.png"),
                normalize(efi_cpu) * 255,
            )
            del efi
            torch.cuda.empty_cache()
        else:
            for i in range(int(num_slice)):

                z = self.zmin + i * self.interval
                recon_slice = normalize(self.reconstruction_angular_spectrum(z)) * 255
                if self.ifsaveslice:
                    recon_slice_cpu = recon_slice.cpu().detach().numpy()
                    cv2.imwrite(
                        os.path.join(
                            self.save_root,
                            "recon_slice_{:0>4d}.png".format(i),
                        ),
                        recon_slice_cpu,
                    )

                del recon_slice
                torch.cuda.empty_cache()
        del self.rawholo
        del self.theta
        del self.fftbox

        torch.cuda.empty_cache()
        write_json(self.write_json_root, {"bar_update": True})

        print("finished")


class Reconstruction_batch:

    def __init__(
        self,
        recon_batch_kwargs,
    ):
        super().__init__()
        self.single_holo = recon_batch_kwargs["single_holo"]
        self.rawholoroot = recon_batch_kwargs["rawholoroot"]
        self.zmin = recon_batch_kwargs["zmin"]
        self.zmax = recon_batch_kwargs["zmax"]
        self.interval = recon_batch_kwargs["interval"]
        self.lamda = recon_batch_kwargs["lamda"]
        self.dx = recon_batch_kwargs["dx"]
        self.Nx = recon_batch_kwargs["Nx"]
        self.Ny = recon_batch_kwargs["Ny"]
        self.device = recon_batch_kwargs["device"]
        self.ifEFI = recon_batch_kwargs["ifEFI"]
        self.ifsaveslice = recon_batch_kwargs["ifsaveslice"]
        self.save_root = recon_batch_kwargs["save_root"]
        self.type = recon_batch_kwargs["type"]

        if self.type == True:
            self.type = "offaxis"
        else:
            self.type = "inline"

        self.thread_kwargs = {}
        self.thread_kwargs["rawholoroot"] = self.rawholoroot
        self.thread_kwargs["zmin"] = self.zmin
        self.thread_kwargs["zmax"] = self.zmax
        self.thread_kwargs["interval"] = self.interval
        self.thread_kwargs["lamda"] = self.lamda
        self.thread_kwargs["dx"] = self.dx
        self.thread_kwargs["Nx"] = self.Nx
        self.thread_kwargs["Ny"] = self.Ny
        self.thread_kwargs["device"] = self.device
        self.thread_kwargs["ifEFI"] = self.ifEFI
        self.thread_kwargs["ifsaveslice"] = self.ifsaveslice
        self.thread_kwargs["save_root"] = self.save_root
        self.thread_kwargs["type"] = self.type

        self.count_finished_thread = 0
        self.write_json_root = r"logging\logging_recon.json"

    def run(self):
        write_json(self.write_json_root, {"msg": "重建类型：" + self.type})

        if self.single_holo != -1 and self.rawholoroot == -1:
            self.single_holo = cv2.imread(self.single_holo, 0)
            self.single_holo = self.single_holo.astype(np.float32)
            self.single_holo = torch.tensor(self.single_holo).to(self.device)
            self.thread_kwargs["rawholo"] = self.single_holo
            self.thread_kwargs["theta"] = False
            self.thread_kwargs["auto_with_bbox"] = False
            worker = Reconstruction_thread(self.thread_kwargs)
            worker.run()
            write_json(self.write_json_root, {"finished": True})

        elif self.single_holo == -1 and self.rawholoroot != -1:
            rawhololist = os.listdir(self.rawholoroot)
            if self.type == "offaxis":
                write_json(self.write_json_root, {"bar_clear": True})
                write_json(self.write_json_root, {"bar_ini": [0, len(rawhololist)]})
                self.single_holo = cv2.imread(
                    os.path.join(self.rawholoroot, rawhololist[0]), 0
                )
                # print(rawhololist[0])
                self.single_holo = self.single_holo.astype(np.float32)
                self.single_holo = torch.tensor(self.single_holo).to(self.device)
                self.thread_kwargs["rawholo"] = self.single_holo
                self.thread_kwargs["theta"] = False
                self.thread_kwargs["auto_with_bbox"] = False
                # if ".tiff" in rawhololist[0]:
                #     name = rawhololist[0][:-5]
                # else:
                #     name = rawhololist[0][:-4]
                # self.thread_kwargs["save_root"] = os.path.join(self.save_root, name)
                worker = Reconstruction_thread(self.thread_kwargs)
                theta, box = worker.cal_offline_angle()
                self.thread_kwargs["theta"] = theta
                self.thread_kwargs["auto_with_bbox"] = box
                # worker.run()

                # write_json(self.write_json_root, {"bar_update": True})
                # print(theta, box)

            else:
                self.thread_kwargs["theta"] = False
                self.thread_kwargs["auto_with_bbox"] = False

            self.processpool = Pool(processes=2)

            with self.processpool as pool:
                for i in rawhololist:
                    print(i)
                    if ".tiff" in i:
                        name = i[:-5]
                    else:
                        name = i[:-4]
                    rawholo = cv2.imread(os.path.join(self.rawholoroot, i), 0)
                    rawholo = rawholo.astype(np.float32)
                    rawholo = torch.tensor(rawholo).to(self.device)
                    self.thread_kwargs["rawholo"] = rawholo
                    self.thread_kwargs["save_root"] = os.path.join(self.save_root, name)
                    print(self.thread_kwargs["theta"])
                    # print(self.thread_kwargs["rawholo"])

                    try:
                        os.mkdir(self.thread_kwargs["save_root"])
                    except:
                        pass
                    worker = Reconstruction_thread(self.thread_kwargs)

                    # self.processpool.apply_async(worker.run, callback=callback_with_self)
                    pool.map(working, [worker])
                    # pool.apply_async(worker.run, callback=callback_with_self)

                # self.processpool.close()
                # self.processpool.join()
            if self.count_finished_thread == len(rawhololist):

                write_json(self.write_json_root, {"finished": True})


def working(worker):
    worker.run()


def main(kwargs):
    worker = Reconstruction_batch(kwargs)
    worker.run()
