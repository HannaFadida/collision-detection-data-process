import boto
from copy import deepcopy
import os
import numpy as np
from scipy.fftpack import fft
from scipy.interpolate import interp1d
import shutil

BUCKET_NAME = 'nexar-artifacts'

fftSpan = 2.0

accSepcturalSamples = 100
gyroSepcturalSamples = 100
gpsSepcturalSamples = 2
magSepcturalSamples = 30

ACC_FILE_NAME = "acc.log"
GYRO_FILE_NAME = "gyro.log"
GPS_FILE_NAME = "gps.log"
MAG_FILE_NAME = "magnetometer.log"

training_per = 0.8

def parse_sensor_file (dataDir, fileName, sensorSepcturalSamples):
    curSensorList = []
    curTimeList = []

    fileIn = open(os.path.join(dataDir, fileName))
    line = fileIn.readline()
    while len(line) > 0:
        elems = line.split(',')
        curTime = float(elems[0])
        sensorData = map(float, elems[1:])

        curSensorList.append(deepcopy(sensorData))
        curTimeList.append(curTime)

        line = fileIn.readline()

    if (len(curSensorList) < 2):
        return None, False

    curSensorListOrg = np.array(curSensorList).T
    curTimeListOrg = np.array(curTimeList)

    curSensorList = curSensorListOrg + 0.
    curTimeList = curTimeListOrg + 0.

    if curTimeList[-1] < fftSpan:
        curTimeList[-1] = fftSpan
    if curTimeList[0] > 0.:
        curTimeList[0] = 0.

    sensorInterp = interp1d(curTimeList, curSensorList)
    sensorInterpTime = np.linspace(0.0, fftSpan * 1, sensorSepcturalSamples * 1)
    sensorInterpVal = sensorInterp(sensorInterpTime)

    sensorFFT = fft(sensorInterpVal).T
    sensorFFTSamp = sensorFFT[::1] / float(1)
    sensorFFTFin = []
    for sensorFFTElem in sensorFFTSamp:
        for axisElem in sensorFFTElem:
            sensorFFTFin.append(axisElem.real)
            sensorFFTFin.append(axisElem.imag)

    return sensorFFTFin, True

s3 = boto.connect_s3()

for datasetType in ['HardBrake']: #['collision', 'HardBrake']:
    train_ids = []
    eval_ids = []

    dataDir = 'collision-detection-sensors-data/' + datasetType

    i = 0
    bucket = s3.get_bucket(BUCKET_NAME)

    #download the whole data first! - iterate over collision-detection-sensors-data

    for incidentId in next(os.walk(dataDir))[1]:
        if incidentId <= "108c05fa4b51d9c3e28c6495662e19ac": continue
        print "preparing " + datasetType + " data " + "with incidentId = " + incidentId
        incidentDir = os.path.join(dataDir, incidentId)


        if not (os.path.exists(os.path.join(incidentDir, "acc.log")) and
                os.path.exists(os.path.join(incidentDir, "gyro.log")) and
                os.path.exists(os.path.join(incidentDir, "gps.log")) and
                os.path.exists(os.path.join(incidentDir, "magnetometer.log"))): continue

        curSenData = []

        accFFTFin, accRes = parse_sensor_file(incidentDir, 'acc.log', accSepcturalSamples)
        gyroFFTFin, gyroRes = parse_sensor_file(incidentDir, 'gyro.log', gyroSepcturalSamples)
        gpsFFTFin, gpsRes = parse_sensor_file(incidentDir, 'gps.log', gpsSepcturalSamples)
        magFFTFin, magRes = parse_sensor_file(incidentDir, 'magnetometer.log', magSepcturalSamples)

        if not (accRes and gyroRes and gpsRes and magRes):
            continue

        curSenData += accFFTFin
        curSenData += gyroFFTFin
        curSenData += gpsFFTFin
        curSenData += magFFTFin
        if datasetType == 'collision':
            curSenData += '1' #collision
        else:
            curSenData += '0'  #HardBrake

        fileOut = open(os.path.join(incidentDir, incidentId + '.csv'), 'w')
        curOut = [str(ele) for ele in curSenData]
        curOut = ','.join(curOut) + '\n'
        fileOut.write(curOut)
        fileOut.close()

        if i%10 < 10 * training_per:
            key = bucket.new_key('collision-detection-dataset/v0/train/train_' + incidentId + '.csv')
            key.set_contents_from_filename(os.path.join(incidentDir, incidentId + '.csv'))
            train_ids.append(incidentId)
        else:
            key = bucket.new_key('collision-detection-dataset/v0/eval/eval_' + incidentId + '.csv')
            key.set_contents_from_filename(os.path.join(incidentDir, incidentId + '.csv'))
            eval_ids.append(incidentId)

        if os.path.exists(incidentDir):
            shutil.rmtree(incidentDir)

        print "preparing " + "incidentId = " + incidentId  + " were completed"
        i += 1

    if not os.path.exists(dataDir):
        os.makedirs(dataDir)

    trainPath = os.path.join(dataDir, datasetType + 'TrainIds.csv')
    trainIdsFile = open(trainPath, 'w')
    trainIdsFile.write('\n'.join(train_ids))
    trainIdsFile.close()

    evalPath = os.path.join(dataDir, datasetType + 'EvalIds.csv')
    evalIdsFile = open(evalPath, 'w')
    evalIdsFile.write('\n'.join(eval_ids))
    evalIdsFile.close()

    key = bucket.new_key("collision-detection-dataset/v0/" + datasetType + "/TrainIds.csv" )
    key.set_contents_from_filename(trainPath)

    key = bucket.new_key("collision-detection-dataset/v0/" + datasetType + "/EvalIds.csv")
    key.set_contents_from_filename(evalPath)

#ACC = 3x2x100
#GYRO = 3x2x100
#GPS = 7x2x2
#MAG = 3x2x30




