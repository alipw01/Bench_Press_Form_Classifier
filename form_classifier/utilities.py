import random
import string
import struct
import tkMessageBox
from Tkinter import *
import serial
import time
import csv
import pyqtgraph as pg
from connectivity_modules.wifi_conn import WifiConnectivity
from connectivity_modules.ble_conn import BleConnectivity
from connectivity_modules.usb_conn import UsbConnectivity


"""
This code contains some of the functionality required to record and store the information obtained by the sensors and is written by the sensor's manufacturers (Inertial Elements), who's work can be accessed below:
URL: https://inertialelements.com/support.html 
"""


pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
isRunning = True

plot_descs = {
    1: ["ACC", 'a', '[m/s^2]'],
    2: ["GYRO", "\omega", ' [deg/s]'],
    3: ["MAG", "tesla", ' [mictes]']
}
app = None


def get_connectivity(c_type):
    """
    Function for obtaining the connection type from the device to the sensor
    param c_type: the connection type
    """
    if str(c_type).upper() == 'WIFI':
        return WifiConnectivity()
    elif str(c_type).upper() == 'BLE':
        return BleConnectivity()
    elif str(c_type).upper() == 'USB':
        return UsbConnectivity()
    return None


def get_rate_divider(a):
    """
    Function for calculating the rate divider value
    """
    if int(a) == 1:
        return 0
    else:
        return 1 + get_rate_divider(a / 2)


def get_checksum(pkt):
    """
    Function for returning the checksum of a packet
    """
    return struct.unpack("!H", pkt[-2:])[0]


def cal_checksum(pkt):
    """
    Function for calculating checksum of packet
    """
    checksum = 0
    a = pkt[:-2].encode('hex')
    x = 0
    y = 2
    for i in range(0, len(a)/2):
        checksum += int(a[x:y], 16)
        x += 2
        y += 2
    return checksum


def convert_bytes_to_string(byte_array):
    """
    Function for converting bytes to a string
    param byte_array: the byte array
    return: the byte array as represented by a string
    """
    binary_str = ''
    for byte in byte_array:
        binary_str += '{0:08b}'.format(byte)
    return binary_str


def id_generator(size=6, chars=string.ascii_lowercase + string.digits):
    """
    Function for generating random ID
    """
    return ''.join(random.choice(chars) for _ in range(size))


def create_logfile(binary_string, select_acm, dlogfile, is_timestamp = True):  
    """
    Function for creating the .csv file used during data collection
    """
    filedata = open(dlogfile, "wb")
    data_fmt = ['Packet No.']
    if is_timestamp:
        data_fmt.append("TimeStamp")

    for indx in range(0, len(binary_string)):
        if binary_string[indx] == '1':
            acc_gyro = select_acm[indx]
            for j in range(0, len(acc_gyro)):
                if acc_gyro[j] == '1':
                    axis = j % 3
                    if j < 3:
                        data_fmt.append("a" + str(axis) + str(indx)+" m/s^2")
                    elif j < 6:
                        data_fmt.append("g" + str(axis) + str(indx) + " deg/s")
                    else:
                        data_fmt.append("m" + str(axis) + str(indx) + " mtesla")

    writer = csv.writer(filedata)
    writer.writerow(data_fmt)
    return filedata


def get_plot_options(select_acm, binary_string):
    """
     Get Plot options
     :param select_acm:
     :param binary_string:
     :return list of number e.g. [1, 2 , 3 ] mean Plot must
            contains Acc(1), Gyro (2) and Magn(3):
    """
    plot_opt = set()
    for indx in range(0, len(binary_string)):
        if binary_string[indx] == '1':
            acc_gyro = select_acm[indx]
            for j in range(0, len(acc_gyro)):
                if acc_gyro[j] == '1':
                    if j < 3:
                        plot_opt.add(1)
                    elif j < 6:
                        plot_opt.add(2)
                    else:
                        plot_opt.add(3)
    return list(plot_opt)


def cal_checksum_cmd(cmd):
    """
    Function for calculating checksum
    """
    checksum = 0
    for i in range(0, len(cmd)):
        checksum += cmd[i]
    return int(checksum / 256), checksum % 256


def open_device(device, rate):
    """
    Function for opening a com port
    """
    try:
        btserial = serial.Serial(device, rate)
    except serial.SerialException as e:  # if it gives error then close the program
        root = Tk()
        root.withdraw()
        tkMessageBox.showerror("Error !", "%s\n\n Please restart Com port and the deivce" % e.message)
        stop = file("error", 'w')
        stop.close()
        sys.exit(1)
    return btserial


def get_ratedevider(a):
    """
    Function for getting the rate divider value
    """
    if int(a) == 1:
        return 0
    else:
        return 1 + get_ratedevider(a/2)


def write_device(device, buffer_c):  
    """
    Function for writing commnads in the com port for the sensor
    """
    device.write(buffer_c)
    device.flushOutput()


def read_device(device, data_len):
    """
    Function for reading the packets coming from the sensor
    """
    buffer_r = device.read(data_len)
    return buffer_r


def is_hex(s):
    """
    Function for checking if s is a hexadecimal or not
    """
    hex_digits = set(string.hexdigits)
    # if s is long, then it is faster to check against a set
    return all(c in hex_digits for c in s)
