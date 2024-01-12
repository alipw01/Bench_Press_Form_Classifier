import os.path
import binascii
import keyboard
from utilities import *


"""
Class for recording the inertial data from the MIMU22BL sensor
This code is an adaptation of the code written by the sensor's manufacturers (Inertial Elements), who's work can be accessed below:
URL: https://inertialelements.com/support.html 
"""
class Mimu22bl:

    def __init__(self, conn_params, queue):
        """
        param conn_params: the connection parameters for the sensor
        param queue: the queue used to contain all recorded data
        """
        self.MAX_FREQ = 250  # Max data rate transmission
        self.g_value = 9.81  # gravity value set
        self.out_rate = float(250)  # Data rate (Hz) Note: Maximum supported data rate is 562.5 Hz when self.plot_graph is set to 0,
                                                        # i.e. when output data is logged in file only, without any plot
        self.conn_params = conn_params # the connection parameters
        self.conn_type = 'usb' # the connection type (functionality is designed for 'usb')

        self.NUM_AXIS = 9  # number of axis e.g ax, ay, az, gx, gy, gz , mx, my and mz
        self.FUSE_ITEM = 1  # number of fused item e.g. timestamp
        self.select_acm = '111111000' # selects accelerometer and gyroscope axes
        self.runtime = float(2.5)  # Enter time in second
        self.binary_string = '1'
        self.num_of_selected_imu = 1 # a single IMU is used
        self.queue = queue # defines the data queue


    def get_inertial_data(self, pkt_d):
        """
        return: the unpacked inertial data packet
        """
        return struct.unpack('!L9f', pkt_d[4:-2])


    def get_queue(self):
        """
        return: the queue containing the inertial data
        """
        return self.queue


    def run(self):
        """
        Function for running the data recording process for the MIMU22BL sensor
        """
        if os.path.isfile("stop"):
            os.remove("stop")

        self.connectivity_obj = get_connectivity(self.conn_type)

        if self.connectivity_obj is None:
            tkMessageBox.showerror("Alert", "%s\nPlease give input connectivity type e.g. USB or WiFi or  BLE")
            sys.exit(1)

        # Open serial port
        try:
            self.com = self.connectivity_obj.open(self.conn_params)
        except Exception as e:
            tkMessageBox.showerror("oops", "%s\nPlease restart the device and com port and try again" % e.message)
            sys.exit(1)

        # generating the command to start normal imu
        out_rate_ = self.MAX_FREQ / float(self.out_rate)
        hex_val = [0x01, 0x02, 0x3, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f]
        out_rate_cmd = hex_val[get_ratedevider(out_rate_)]
        checksum = 0x39 + 0x40 + out_rate_cmd
        self.cmd = [0x39, 0x40 + out_rate_cmd, 0x00, checksum]
        print('Mimu22BL Ready. Once all Sensors are ready, press *Space Bar* to begin. The Process can be stopped by pressing the "q" key.')
        # waits until *space bar* is pressed before starting the command to start the sensor recording
        # this is done to synchronise the two sensors
        while keyboard.is_pressed(" ") == False:
            time.sleep(0.001)
        self.connectivity_obj.send(self.cmd)

        self.pkt_number = []
        scale_pr_acc = 1.0
        scale_pr_gyro = 57.325
        scale_pr_mag = 0.3
        self.pkts = 0
        count = 0

        s1 = ''
        pkt_size = 46
        # receives initial data packet
        Data_in = self.connectivity_obj.receive(pkt_size)
        self.start = time.time()

        # keeps obtaining new data until 'q' is pressed to terminate it
        while keyboard.is_pressed("q") == False:
            if not os.path.isfile("stop"):
                try:
                    Data_in = binascii.unhexlify(Data_in)
                    s1 = Data_in.encode("hex")
                    (start_code, pkt_num, payload_length) = struct.unpack("!BHB", Data_in[0:4])
                    if struct.unpack("!B", Data_in[0])[0] == 0xAA and get_checksum(Data_in) == cal_checksum(Data_in):
                        values = self.get_inertial_data(Data_in)
                        self.pkt_number.append(pkt_num)
                        # putting the data into lists
                        data = [pkt_num, time.time(), round(values[1]*scale_pr_acc, 3),
                                round(values[2]*scale_pr_acc, 3), round(values[3]*scale_pr_acc, 3),
                                round(values[4]*scale_pr_gyro, 3), round(values[5]*scale_pr_gyro, 3),
                                round(values[6]*scale_pr_gyro, 3), round(values[7]*scale_pr_mag, 3),
                                round(values[8]*scale_pr_mag, 3), round(values[9]*scale_pr_mag, 3)]
                        # put data in queue
                        self.queue.put(data)
                        # receive new data packet
                        Data_in = self.connectivity_obj.receive(pkt_size)
                        self.pkts += 1
                        count = 0

                    elif re.search(b'[\d|\w]+aa.*', s1):  # search and find new packet
                        lst = re.findall(b'[\d|\w]+(aa.*)', s1)
                        str_rem = lst[0]
                        length = len(str_rem) / 2
                        pkt_rem = Data_in[-length:]
                        new_len = pkt_size - length
                        Data_in = self.connectivity_obj.receive(new_len)
                        Data_in = pkt_rem.encode("hex") + Data_in

                    else:
                        Data_in = self.connectivity_obj.receive(pkt_size)
                        # exit the code if the packet is detecting wrong continuously for more than 5 times
                        count += 1
                        if count > 5:
                            count = 0

                            tkMessageBox.showinfo("Oops",
                                                "Something went wrong please restart the device and run the process again !")

                            self.com.close()
                            sys.exit(1)
                except TypeError as e:
                    s2 = ''
                    for chk in range(0, len(Data_in)):
                        if is_hex(Data_in[chk]) is True:
                            s2 += Data_in[chk]
                    if re.search(b'[\d|\w]+aa.*', s2):  # search and find new packet
                        lst = re.findall(b'[\d|\w]+(aa.*)', s2)
                        str_rem = lst[0]
                        length = len(str_rem) / 1
                        pkt_rem = Data_in[-length:]
                        new_len = pkt_size * 2 - length
                        Data_in = self.connectivity_obj.receive(new_len)
                        Data_in = pkt_rem + Data_in
                    else:
                        Data_in = self.connectivity_obj.receive(pkt_size)
                        # exit the code if the packet is detecting wrong continuously for more than 5 times
                        count += 1
                        if count > 5:
                            tkMessageBox.showinfo("Oops",
                                                "Something went wrong please restart the device and run the process again !")
                            sys.exit(1)
                except KeyboardInterrupt:
                    print ("Error")
                    cmd = [0x32, 0x00, 0x32]
                    self.connectivity_obj.send(cmd)
                    cmd = [0x22, 0x00, 0x22]
                    self.connectivity_obj.send(cmd)
                    self.connectivity_obj.close()
                    sys.exit(1)
            else:
                self.connectivity_obj.send([0x32, 0x00, 0x32])
                self.connectivity_obj.send([0x22, 0x00, 0x22])
                self.com.close()
                break
        
        try:  # exit for when 'q' is pressed and code is terminated
            self.connectivity_obj.send([0x32, 0x00, 0x32])
            self.connectivity_obj.send([0x22, 0x00, 0x22])
        except Exception:
            pass

        isRunning = False
        stop = time.time()
        if os.path.isfile("stop"):
            os.remove("stop")