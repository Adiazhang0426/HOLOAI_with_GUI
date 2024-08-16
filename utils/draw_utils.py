"""
@Description: 
@Author: Adiazhang
@Date: 2024-06-04 09:56:06
@LastEditTime: 2024-06-04 10:45:29
@LastEditors: Adiazhang
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import pandas as pd


def is_nested(lst):
    for element in lst:
        if isinstance(element, list):
            return True
    return False


class Draw_Save:
    """
    @Description:
    @param {*} self
    @param {*} particle
    @param {*} pixel cm
    @param {*} deltaz
    @param {*} W
    @param {*} H
    @param {*} image_number
    @param {*} bin_interval
    @return {*}
    """

    def __init__(
        self,
        kwargs,
    ):
        super().__init__()
        # ini particles are [[[xyzd],[xyzd]],[]]
        self.particle = kwargs["particle"]
        self.minz = kwargs["minz"]
        self.deltaz = kwargs["deltaz"]
        self.pixel = kwargs["pixel"]
        self.W = kwargs["W"]
        self.H = kwargs["H"]
        self.bin_interval = kwargs["bin_interval"]
        self.save_path = kwargs["save_path"]
        self.method_type = kwargs["method_type"]

        self.pp_config = kwargs["pp_config"]
        # self.image_number=kwargs['image_number']
        self.img_name_list = kwargs["img_name_list"]

    def pre_process(self):
        if is_nested(self.particle):
            new_particle = []
            new_split_particle = []
            for pars in self.particle:
                new_temp_split_particle = []
                for par in pars:
                    par[0], par[1], par[3] = (
                        par[0] * self.pixel,
                        par[1] * self.pixel,
                        par[2] * self.pixel,
                    )
                    par[2] = par[2] * self.deltaz + self.minz
                    new_particle.append(par)
                    new_temp_split_particle.append(par)
                new_split_particle.append(new_temp_split_particle)
            self.split_particle = new_split_particle
            self.particle = new_particle
        else:
            self.split_particle = None
        # allcase = os.listdir(self.save_path)
        # ini_case_id = 0
        # while "case{}".format(ini_case_id) in allcase:
        #     ini_case_id += 1
        # self.save_path = os.path.join(self.save_path, "case{}".format(ini_case_id))
        # os.makedirs(self.save_path)
        self.cal_volumn()
        self.lwc = self.cal_lwc()
        self.mvd = self.cal_mvd()
        self.volume_ratio, self.cul_volume_ratio, self.bins, self.binlist = (
            self._cal_dia_volumn_ratio()
        )

    def cal_parameter(self):
        self.cal_volumn()
        self.lwc = self.cal_lwc()
        self.mvd = self.cal_mvd()
        self.volume_ratio, self.cul_volume_ratio, self.bins, self.binlist = (
            self._cal_dia_volumn_ratio()
        )

    def cal_volumn(self):
        water = []
        for i in self.particle:
            d = i[-1]
            water.append(np.pi * 4 / 3 * (d / 2) ** 3)
        self.volume = sum(water)
        self.water = water

        return self.volume, self.water

    """
    @Description: LWC calculation g/cm3
    @param {*} particle same as cal_volumn
    @param {*} pixel
    @param {*} deltaz the length of the measurement volume in z axis
    @param {*} W
    @param {*} H
    @param {*} image_number number of images in the measurement volume
    @return {*}
    """

    def cal_lwc(self):

        return (
            self.volume
            / (self.W * self.pixel)
            / (self.H * self.pixel)
            / self.deltaz
            / len(self.img_name_list)
            * (1e3)
        )

    def cal_mvd(self):
        """
        中值体积直径
        """

        water = sorted(self.water)
        allvolumn = self.volume / 2
        v_sum = 0
        for i in range(len(water)):
            v_sum += water[i]
            if v_sum - water[i - 1] < allvolumn and v_sum > allvolumn:
                mvd = (water[i] * 3 / 4 / np.pi) ** (1 / 3) * 2
                break

        return mvd

    """
    @Description: 
    @param {*} alld particle diameter
    @param {*} volumn volume list for all particles
    @param {*} interval the interval of each bin
    @param {*} pixel
    @return {*}
    """

    def _cal_dia_volumn_ratio(self):
        alld = np.array(self.particle)[:, -1]
        alld = alld.tolist()
        mind = min(alld)
        maxd = max(alld)
        bins = int((maxd - mind) / self.bin_interval)
        self.bins = bins
        binlist = [i * self.bin_interval for i in range(bins)]
        allvolumn = self.volume
        volumn_ratio = []
        for i in range(bins):
            single_sum = 0
            sd = i * self.bin_interval
            bd = (i + 1) * self.bin_interval
            sv = 4 / 3 * np.pi * (sd / 2) ** 3
            bv = 4 / 3 * np.pi * (bd / 2) ** 3

            for j in self.water:
                if sv <= j <= bv:
                    single_sum += j
            volumn_ratio.append(single_sum / allvolumn)
        cumulative_volumn = []
        for i in range(1, len(volumn_ratio) + 1):
            cumulative_volumn.append(sum(volumn_ratio[:i]))
        return volumn_ratio, cumulative_volumn, self.bins, binlist

    def draw_3D_distribution(self):
        """
        @Description: 绘制三维分布图
        @param {*}
        @return {*}
        """
        particle = np.array(self.particle)
        x = particle[:, 0]
        y = particle[:, 1]
        z = particle[:, 2]
        d = particle[:, 3]
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(x, y, z, c=d, marker="o")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.colorbar()
        ax.title("3D Distribution")
        plt.savefig(os.path.join(self.save_path, "3D_distribution.png"))

    def draw_dia_distribution(self):
        """
        @Description: 绘制直径分布图
        @param {*}
        @return {*}
        """
        particle = np.array(self.particle)
        d = particle[:, 3]
        plt.hist(d, bins=self.bins, density=True)
        plt.xlabel("Diameter")
        plt.ylabel("Frequency")
        plt.title("Diameter Distribution")
        plt.savefig(os.path.join(self.save_path, "dia_distribution.png"))

    def draw_volumn_distribution(self):
        plt.subplot(1, 2, 1)
        plt.bar(self.binlist, self.cul_volume_ratio, width=2)
        plt.xlabel("Diameter")
        plt.ylabel("Culmative Volume Ratio")
        plt.title("Culmative Volume Ratio")
        plt.subplot(1, 2, 2)
        plt.bar(self.binlist, self.volume_ratio, width=2)
        plt.xlabel("Diameter")
        plt.ylabel("Volume Ratio")
        plt.title("Volume Ratio")
        plt.savefig(os.path.join(self.save_path, "volumn_distribution.png"))

    def write_result(self):
        with open(os.path.join(self.save_path, "config.txt"), "w") as f:
            f.write("deltaz {}".format(self.deltaz) + "\n")
            f.write("num_image {}".format(len(self.img_name_list)) + "\n")
            f.write("H {}".format(self.H) + "\n")
            f.write("W {}".format(self.W) + "\n")
            f.write("pixel {}".format(self.pixel) + "\n")
        with open(os.path.join(self.save_path, "result.txt"), "w") as f:
            f.write("Method type " + self.method_type + "\n")
            f.write("Post process config: \n")
            for key, value in self.pp_config.items():
                f.write(key + ": " + str(value) + "\n")
            f.write("X Y Z D" + "\n")
            for particles, image_name in zip(self.split_particle, self.img_name_list):
                f.write("Results for" + image_name + "\n")
                for particle in particles:
                    f.write(
                        str(particle[0])
                        + " "
                        + str(particle[1])
                        + " "
                        + str(particle[2])
                        + " "
                        + str(particle[3])
                        + "\n"
                    )
                f.write("Particle number: " + str(len(particles)) + "\n")
            f.write("Total article number: " + str(len(self.particle)) + "\n")
            f.write("LWC: {} MVD: {}".format(self.lwc, self.mvd))

        particle = np.array(self.particle)
        particle = particle.T
        df = pd.DataFrame(particle)
        df.to_csv(
            os.path.join(self.save_path, "all_result.csv"),
            index=False,
            header=["X", "Y", "Z", "D"],
        )
        for particles, image_name in zip(self.split_particle, self.img_name_list):
            split_par = np.array(particles)
            split_par = split_par.T
            df = pd.DataFrame(split_par)
            df.to_csv(
                os.path.join(self.save_path, image_name + "_result.csv"),
                index=False,
                header=["X", "Y", "Z", "D"],
            )

    def run(self):
        """
        @Description: 运行
        @param {*}
        @return {*}
        """
        self.pre_process()
        self.draw_3D_distribution()
        self.draw_dia_distribution()
        self.draw_volumn_distribution()
        self.write_result()
