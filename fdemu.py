import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Sensor Emulator',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--mode', default='FD', type=str,
                    help='Sensor mode' )

def bound(low, high, value):
    return max(low, min(high, value))


class FDemu:
    """docstring for ClassName"""
    def __init__(self, iniframe):
        self.prefm = iniframe
        self.th = 10
        self.choicelist=[np.uint8(0), np.uint8(3)]
        self.sdmaxidx = 0

        self.sdcount = 0
        self.sdc_pre = 0
        self.sdc_th = 6
        self.isTracking = False
        self.bkloc = np.array([0, 0])

        self.salien_th = 3

        self.srchx = 4
        self.srchy = 4

        self.bksalience = 0.0
        self.bmsali = 0.0

    def FDmode(self, inputfm):
        self.currtfm = inputfm
        self.diff = self.currtfm.astype(int) - self.prefm.astype(int)
        # print (self.diff.dtype)
        self.prefm = inputfm
        self.condlist = [self.diff > self.th, self.diff < -self.th]
        self.output = np.select(self.condlist,self.choicelist,default=np.uint8(2))
        return self.output
    def SDmode(self, inputfm):
        self.FDmode(inputfm)
        self.SDmatrix = np.zeros((8, 8), dtype=np.uint8)
        self.SDoutput = np.where(self.output != 2, 1, 0)
        for i in range(8):
            for j in range(8):
                self.SDmatrix[i,j] = np.sum(self.SDoutput[i*8:i*8+8,j*8:j*8+8])
        self.maxval = np.argmax(self.SDmatrix)
        self.maxidx = (self.maxval%8, self.maxval//8)
        return self.maxidx
    def BlockMatch(self, inputfm):
        self.SDmode(inputfm)

        if (self.sdc_pre == self.maxidx):
            self.sdcount += 1
        else:
            self.sdcount = 0
        self.sdc_pre = self.maxidx
        self.initbkloc = np.asarray(self.maxidx)*8

        if ((self.sdcount>self.sdc_th) & (self.isTracking == False)):
            self.isTracking = True
            self.bkloc = self.initbkloc
            self.bmsali = 10

        if (self.isTracking == True):
            # Run Block Matching

            bsrhx_u = bound(0, 64-8, self.bkloc[1]-self.srchx)
            bsrhx_d = bound(0, 64-8, self.bkloc[1]+self.srchx)
            bsrhy_u = bound(0, 64-8, self.bkloc[0]-self.srchy)
            bsrhy_d = bound(0, 64-8, self.bkloc[0]+self.srchy)
            # print (bsrhx_u, bsrhx_d, bsrhy_u, bsrhy_d)
            self.SADmax = 0
            for i in range(bsrhx_u, bsrhx_d):
                for j in range(bsrhy_u, bsrhy_d):
                    SADmatrix = self.output[self.bkloc[1]:self.bkloc[1]+8, self.bkloc[0]:self.bkloc[0]+8].astype(int) - self.output[i:i+8, j:j+8].astype(int)
                    SAD = np.sum(np.abs(SADmatrix))
                    if (self.SADmax < SAD):
                        self.SADmax = SAD
                        self.bmindx = (i,j)
                    # print ('SAD')
                    # print (SAD)
                    
                    # print (self.isTracking)
            self.bkloc = np.array([self.bmindx[1], self.bmindx[0]])
            self.bksalience = np.sum(self.SDoutput[self.bkloc[1]:self.bkloc[1]+8, self.bkloc[0]:self.bkloc[0]+8])
            self.bmsali = self.bmsali/2 + self.bksalience

            # if SAD too small
            if (self.bmsali < self.salien_th):

                self.isTracking = False

def main():
    global args
    args = parser.parse_args()

    fdemu = FDemu(np.zeros((64, 64), dtype=np.uint8))
    # define a video capture object
    vid = cv2.VideoCapture(0)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    while(True):
      
        # Capture the video frame
        # by frame
        ret, frame = vid.read()

        crop_frame = frame[:, 260:980]
        img2 = cv2.resize(cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY), (64, 64), interpolation=cv2.INTER_LINEAR)


        # Display the resulting frame
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
        cv2.resizeWindow("frame", 640, 640)
        cv2.imshow("frame", img2)


        if (args.mode == 'FD'):
            fdframe = fdemu.FDmode(img2)
            fdframe *= np.uint8(85)
            cv2.namedWindow("FDframe", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
            cv2.resizeWindow("FDframe", 640, 640)
            cv2.imshow("FDframe", fdframe)

        elif (args.mode == 'SD'):
            maxidx = fdemu.SDmode(img2)
            # maxidx = (maxval%8, maxval//8)
            fdframe = cv2.cvtColor((fdemu.output)*np.uint8(85), cv2.COLOR_GRAY2BGR)
            shapes = np.zeros_like(fdframe, np.uint8)
            # Draw shapes
            cv2.rectangle(shapes, (maxidx[0]*8, maxidx[1]*8), (maxidx[0]*8+8, maxidx[1]*8+8), (0, 0, 255), cv2.FILLED)
            mask = shapes.astype(bool)

            # Display the resulting frame
            cv2.namedWindow("frame", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
            cv2.resizeWindow("frame", 640, 640)
            cv2.imshow("frame", img2)


            cv2.namedWindow("FDframe", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
            cv2.resizeWindow("FDframe", 640, 640)
            out = fdframe.copy()
            out[mask] = cv2.addWeighted(fdframe, 0.05, shapes, 0.95, 0)[mask]
            cv2.imshow("FDframe", out)

        elif (args.mode == 'BM'):
            fdemu.BlockMatch(img2)
            fdframe = cv2.cvtColor((fdemu.output)*np.uint8(85), cv2.COLOR_GRAY2BGR)
            shapes = np.zeros_like(fdframe, np.uint8)
            # Draw shapes
            cv2.rectangle(shapes, (fdemu.maxidx[0]*8, fdemu.maxidx[1]*8), (fdemu.maxidx[0]*8+8, fdemu.maxidx[1]*8+8), (0, 0, 255), cv2.FILLED)
            mask = shapes.astype(bool)

            # Display the resulting frame
            cv2.namedWindow("frame", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
            cv2.resizeWindow("frame", 640, 640)
            cv2.imshow("frame", img2)


            cv2.namedWindow("FDframe", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
            cv2.resizeWindow("FDframe", 640, 640)
            out = fdframe.copy()
            out[mask] = cv2.addWeighted(fdframe, 0.05, shapes, 0.95, 0)[mask]
            if (fdemu.isTracking == True):
                out[fdemu.bmindx[0]:fdemu.bmindx[0]+8, fdemu.bmindx[1]] = (0, 255, 0)
                out[fdemu.bmindx[0]:fdemu.bmindx[0]+8, fdemu.bmindx[1]+8] = (0, 255, 0)
                out[fdemu.bmindx[0], fdemu.bmindx[1]:fdemu.bmindx[1]+8] = (0, 255, 0)
                out[fdemu.bmindx[0]+8, fdemu.bmindx[1]:fdemu.bmindx[1]+8] = (0, 255, 0)

            cv2.imshow("FDframe", out)
        
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
  
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()