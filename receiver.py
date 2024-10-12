import serial
import numpy as np
import matplotlib.pylab as plt
import time

ser = serial.Serial('COM19', 115200)

times = 0
itration = 30000

all_data = {
    'rtt_data': [],
    'rssi_data': []
}

rtt_list = []
rssi_list = []
rawFrame = []

start_time = time.time()
while times < itration:
        byte  = ser.read(1)        
        rawFrame += byte
        if rawFrame[-2:]==[13, 10]:
            if len(rawFrame) == 10:
                decimal_data = int.from_bytes(rawFrame[0:4],byteorder='big')
                #print('RTT time:',decimal_data)
                response_rssi = bytes(rawFrame[5:9])
                response_rssi = int(response_rssi.decode('utf-8'))

                #print('request rssi:',request_rssi)
                #print('-------------------------------')
                times = times + 1
                print(times)
                all_data['rtt_data'].append(decimal_data)
                all_data['rssi_data'].append(response_rssi)
                #print('decimal_data:', decimal_data)
                #print('rssi_data:', response_rssi)
            rawFrame = []

end_time = time.time()
print(end_time - start_time)

if len(all_data['rtt_data']) == itration:
    all_data['rtt_data'] = np.array(all_data['rtt_data'])
    all_data['rssi_data'] = np.array(all_data['rssi_data'])

    np.savez('test_data/test_data_distance_1.npz', **all_data)