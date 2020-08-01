import os
import sys
from config import config
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import urllib
from copy import copy
import argparse
from sklearn.cluster import KMeans
import cv2
import numpy as np
import time
import threading

global COLORS
global CURRENT_PLAYING_SONG

def pixel_rgb_to_hsv(pixel):
    rp,gp,bp = np.divide(pixel, 255)
    cmax = max(rp,gp,bp)
    cmin = min(rp, gp, bp)
    delta = cmax-cmin
    
    if delta == 0:
        hue = 0
    elif cmax == rp:
        hue = (60 * (((gp-bp)/delta) % 6))
    elif cmax == gp:
        hue = (60 * (((bp-rp)/delta) + 2))
    elif cmax == bp:
        hue = (60 * (((rp-gp)/delta) + 4))

    if cmax == 0:
        saturation = 0
    else:
        saturation = delta/cmax

    value = cmax
    return [hue,saturation,value]

def pixel_hsv_to_rgb(pixel):
    h,s,v = pixel
    c = v*s
    x = c*(1-abs(((h/60)%2)-1))
    m = v-c

    if h < 60 and h >= 0:
        rp,gp,bp = c,x,0
    elif h < 120 and h >= 60:
        rp,gp,bp = x,c,0
    elif h < 180 and h >= 120:
        rp,gp,bp = 0,c,x
    elif h < 240 and h >= 180:
        rp,gp,bp = 0,x,c
    elif h < 300 and h >= 240:
        rp,gp,bp = x,0,c
    elif h < 360 and h >= 300:
        rp,gp,bp = c,0,x

    r,g,b = (rp+m)*255, (gp+m)*255, (bp+m)*255
    return [r,g,b]

def adjust_light(rgb, new_value):
    assert new_value <= 1 or new_value >= 0
    h,s,v = pixel_rgb_to_hsv(rgb)
    v = new_value
    _rgb = pixel_hsv_to_rgb([h,s,v])
    return _rgb

def fix_saturation(rgb):
    h,s,v = pixel_rgb_to_hsv(rgb) 
    s += s/2
    if s > 1:
        s = 1
    _rgb = pixel_hsv_to_rgb([h,s,v])
    return _rgb

class Animations:
    def __init__(self, led, lock, args=None):
        self.led = led
        self.CROP_SCALE = 24
        self.lock = lock
        self.led_length = 144
        self.args = args
    
    def SoftTrans(self):
        yes=0
        if COLORS == None:
            return -1
        else:
            part_len = (self.led_length-self.CROP_SCALE)//len(COLORS)
            total_time = 0 # should help break when greater than 30 seconds
            time_wanted = 3
            divisor = 0.1

            colors, _colors, tmp_colors, len_colors, changeable_colors = copy(COLORS), copy(COLORS), [i[1] for i in COLORS], len(COLORS), copy(COLORS) #tmp_colors will be used in blending
            if len_colors % 2 != 0:
                # fix middle:
                tmp_colors[(len_colors//2)] = [int(x) for x in colors[len_colors//2][1]]
            degrade_step = 1/(time_wanted/divisor)

            while(True):
                alpha = 1
                beta = 1-alpha

                while(total_time <= time_wanted):
                    if not self.lock.locked():
                        self.lock.acquire(True)
                        try:
                            if COLORS == None or colors != COLORS:
                                # COLOR has changed!!! break transition and return what color we now for BetweenSongTransition
                                print('COLOR HAS CHANGED!')
                                for i in range(len_colors):
                                    _colors[i] = (_colors[i][0], tmp_colors[i])
                                _colors.reverse()
                                return _colors
                        except Exception as e:
                            self.lock.release()
                        finally:
                            if self.lock.locked():
                                self.lock.release()
                    
                    for i in range(len_colors//2):
                        c1 = np.array([int(x) for x in _colors[i][1]], dtype=np.uint8)
                        c2 = np.array([int(x) for x in _colors[len_colors - 1 - i][1]], dtype=np.uint8)
                        c3 = np.zeros((3,), dtype=np.uint8)
                        c4 = np.zeros((3,), dtype=np.uint8)

                        cv2.addWeighted(c1,alpha,c2,beta,0,c3)
                        cv2.addWeighted(c2,alpha,c1,beta,0,c4)
                        tmp_colors[i] = c4
                        tmp_colors[len_colors-1-i] = c3

                    for i,c in enumerate(tmp_colors):
                        _c = [int(x) for x in c]
                        _c = [_c[2], c[1], c[0]]
                        self.led[(i)*part_len:(i+1)*part_len] = [_c for j in range(part_len)]

                    alpha = alpha-degrade_step
                    beta = 1-alpha

                    total_time += divisor
                    if self.args != None:
                        if self.args.dev:
                            self.led.visualize()

                changeable_colors.reverse()
                _colors = changeable_colors
                
                total_time = 0
            return 0

    def BetweenSongTrans(self, new_colors, old_colors):
        print(new_colors)
        
        if new_colors == None:
            print('just resetting')
            self.led.fill((0,0,0))
            #self.led.visualize(waittime=1)
        else:
            part_len = (self.led_length-self.CROP_SCALE)//len(new_colors)
            total_time = 0 # should help break when greater than 30 seconds
            time_wanted = 0.5
            divisor = 0.1

            # take lock!
            self.lock.acquire()
            try:
                colors, _colors, tmp_colors, len_colors = copy(new_colors), copy(new_colors), [i[1] for i in new_colors], len(new_colors) #tmp_colors will be used in blending
            except Exception as e:
                self.lock.release()
            finally:
                if self.lock.locked():
                    self.lock.release()
            
            if len_colors % 2 != 0:
                tmp_colors[(len_colors//2)] = [int(x) for x in colors[len_colors//2][1]]
            
            degrade_step = 1/(time_wanted/divisor)
            alpha = 1
            beta = 1-alpha

            while(total_time <= time_wanted):
                if not self.lock.locked():
                    for i in range(len_colors):
                        # from old_colors
                        c1 = np.array([int(x) for x in old_colors[i][1]], dtype=np.uint8)
                        # from new COLORS
                        c2 = np.array([int(x) for x in _colors[i][1]], dtype=np.uint8)
                        
                        c3 = np.zeros((3,), dtype=np.uint8)
                        c4 = np.zeros((3,), dtype=np.uint8)

                        cv2.addWeighted(c1,alpha,c2,beta,0,c3)
                        cv2.addWeighted(c2,alpha,c1,beta,0,c4)
                        tmp_colors[i] = c4
                        tmp_colors[len_colors-1-i] = c3

                    for i,c in enumerate(tmp_colors):
                        _c = [int(x) for x in c]
                        _c = [_c[2], _c[1], _c[0]]
                        self.led[(i)*part_len:(i+1)*part_len] = [_c for j in range(part_len)]

                    #print(self.led.get_led())
                    #self.led.visualize(waittime=1)
                    
                    alpha = alpha-degrade_step
                    beta = 1-alpha

                    #time.sleep(divisor)
                    total_time += divisor
                else:
                    pass

            return 0

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev', action='store_true', default=False)
    return parser.parse_args()

def parse_current_playing(payload):
    if payload == None:
        return {'image': np.zeros((100,100,3)), 'is_playing': False, 'name':'None'}
    
    try:
        img_path = payload['item']['album']['images'][-1]['url']
        name = payload['item']['name']
        urllib.request.urlretrieve(img_path, './tmp/thumb.png')
        img = cv2.imread('./tmp/thumb.png')
    except Exception as e:
        print(e)
        img = np.zeros((100,100,3))
        name = 'None'
        payload = dict()
        payload['is_playing'] = False

    return {'image': img, 'is_playing': payload['is_playing'], 'name':name}

def led_logic(led, lock, profile=None):
    colors = COLORS
    anim = Animations(led, lock, args=args)

    while(True):
        if not lock.locked():
            colors = COLORS

        if colors != None:
            old_colors = anim.SoftTrans()
            anim.BetweenSongTrans(COLORS, old_colors)
        else:
            led.fill((0,0,0))

def dom_color(img):
    # reshapes into a 2d array of x*y pixels
    reshaped = img.reshape((img.shape[0] * img.shape[1], 3))
    cluster = KMeans(n_clusters=5).fit(reshaped)

    # Get the number of different clusters, create histogram, and normalize
    labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    (hist, _) = np.histogram(cluster.labels_, bins = labels)
    hist = hist.astype("float")
    hist /= hist.sum()
    centroids = cluster.cluster_centers_
    colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)], reverse=True)
    
    # fix less than 5 clusters:
    if len(colors) < 5:
        thing_to_add_at_end = colors[-1]
        for i in range(5-len(colors)):
            colors.append(thing_to_add_at_end)
    
    return colors


if __name__ == '__main__':
    scope = "user-read-currently-playing user-read-playback-state"
    launch_initially = False
    resume_same_song = False
    COLORS = None
    CURRENT_PLAYING_SONG = ''
    _lock = threading.Lock()

    sp = spotipy.Spotify(auth_manager=SpotifyOAuth( \
        client_id=config.SPOTIPY_CLIENT_ID, \
        client_secret=config.SPOTIPY_CLIENT_SECRET, \
        redirect_uri=config.SPOTIPY_REDIRECT_URI, \
        username=config.USERNAME, \
        scope=scope))

    args = parseArguments()
    
    if args.dev:
        from ws2x import LEDSim
        led = LEDSim()
    else:
        import board
        import neopixel
        led = neopixel.NeoPixel(board.D18, 144)
        # force reset whatever was on the LED strip before:
        led.fill((0,0,0))

    while(True):
        try:
            # make sure the playback method goes through successfully:
            raw_output = sp.current_playback()
            _lock.acquire()
            out = parse_current_playing(raw_output)
            _lock.release()
        except Exception as e:
            print(e)
            if _lock.locked():
                _lock.release()

        if out['is_playing'] == False:
            resume_same_song = True
            _lock.acquire(True)
            try:
                COLORS = None
            finally:
                _lock.release()
            
            time.sleep(2)
        else:
            if launch_initially == True or raw_output == None:
                time.sleep(2)
            
            if CURRENT_PLAYING_SONG != out['name'] or resume_same_song:
                resume_same_song = False
                print('generating new colors...')
                
                with _lock:
                    try:
                        COLORS = dom_color(out['image'])

                        # do color corrections here:
                        for i in range(len(COLORS)):
                            tru_color = COLORS[i][1]
                            tru_color = fix_saturation(tru_color)
                            tru_color = adjust_light(tru_color, 0.5)
                            corrected_colors = (COLORS[i][0],)

                    except Exception as e:
                        print(e)
                        print('continuing')
                        COLORS = [(1, (125,125,125))]

                # launch led thread:
                if launch_initially == False:
                    print('Launching LED thread...')
                    _led_thread = threading.Thread(target=led_logic, args=(led,_lock))
                    _led_thread.start()
                    launch_initially = True
                
                CURRENT_PLAYING_SONG = out['name']
