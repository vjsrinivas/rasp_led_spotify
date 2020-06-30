import os
import sys
from config import config
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import urllib
from copy import copy

LEDSTRIP_API = '/home/vijay/Documents/devmk4/LEDStrip/'
sys.path.append(LEDSTRIP_API)
from led import LEDSim

from sklearn.cluster import KMeans
import cv2
import numpy as np
import time
import threading

global COLORS
global CURRENT_PLAYING_SONG

class Animations:
    def __init__(self, led, lock):
        self.led = led
        self.CROP_SCALE = 24
        self.lock = lock

    def SoftTrans(self):
        if COLORS == None:
            return -1
        else:
            part_len = (led.get_length()-self.CROP_SCALE)//len(COLORS)
            total_time = 0 # should help break when greater than 30 seconds
            time_wanted = 10
            divisor = 0.2

            colors, _colors, tmp_colors, len_colors = copy(COLORS), copy(COLORS), [i[1] for i in COLORS], len(COLORS) #tmp_colors will be used in blending
            if len_colors % 2 != 0:
                # fix middle:
                #print('HERE', colors[len_colors//2])
                tmp_colors[(len_colors//2)] = [int(x) for x in colors[len_colors//2][1]]
            degrade_step = 1/(time_wanted/divisor)
            #print(_colors)

            while(True):
                #print('test')
                alpha = 1
                beta = 1-alpha

                while(total_time <= time_wanted):
                    if not self.lock.locked():
                        if colors != COLORS:
                            # COLOR has changed!!! break transition and return what color we now for BetweenSongTransition
                            print('COLOR HAS CHANGED!')
                            for i in range(len_colors):
                                _colors[i] = (_colors[i][0], tmp_colors[i])
                            _colors.reverse()
                            return _colors


                    for i in range(len_colors//2):
                        c1 = np.array([int(x) for x in _colors[i][1]], dtype=np.uint8)
                        c2 = np.array([int(x) for x in _colors[len_colors - 1 - i][1]], dtype=np.uint8)
                        c3 = np.zeros((3,), dtype=np.uint8)
                        c4 = np.zeros((3,), dtype=np.uint8)

                        cv2.addWeighted(c1,alpha,c2,beta,0,c3)
                        cv2.addWeighted(c2,alpha,c1,beta,0,c4)
                        tmp_colors[i] = c4
                        tmp_colors[len_colors-1-i] = c3

                    #print('ORIGINALS:', colors)
                    #print()
                    #print('TMP:',tmp_colors)

                    for i,c in enumerate(tmp_colors):
                        _c = [int(x) for x in c]
                        self.led[(i+1)*part_len:(i+2)*part_len] = [_c for j in range(part_len)]

                    #print(self.led.get_led())
                    self.led.visualize(waittime=1)
                    
                    alpha = alpha-degrade_step
                    beta = 1-alpha

                    time.sleep(divisor)
                    total_time += divisor

                for i in range(len_colors):
                    _colors[i] = (_colors[i][0], tmp_colors[i])
                _colors.reverse()

                total_time = 0

            return 0

    def BetweenSongTrans(self, new_colors, old_colors):
        print(new_colors)
        if new_colors == None:
            print('just resetting')
            self.led.reset()
            #self.led.visualize(waittime=1)
        else:
            part_len = (led.get_length()-self.CROP_SCALE)//len(new_colors)
            total_time = 0 # should help break when greater than 30 seconds
            time_wanted = 0.5
            divisor = 0.1

            colors, _colors, tmp_colors, len_colors = copy(new_colors), copy(new_colors), [i[1] for i in new_colors], len(new_colors) #tmp_colors will be used in blending
            if len_colors % 2 != 0:
                tmp_colors[(len_colors//2)] = [int(x) for x in colors[len_colors//2][1]]
            
            degrade_step = 1/(time_wanted/divisor)
            alpha = 1
            beta = 1-alpha

            while(total_time <= time_wanted):
                print('TRANS')
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
                        self.led[(i+1)*part_len:(i+2)*part_len] = [_c for j in range(part_len)]

                    #print(self.led.get_led())
                    #self.led.visualize(waittime=1)
                    
                    alpha = alpha-degrade_step
                    beta = 1-alpha

                    time.sleep(divisor)
                    total_time += divisor
                else:
                    pass

            return 0


def parse_current_playing(payload):
    if payload == None:
        return {'image': np.zeros((100,100,3)), 'is_playing': False, 'name':'None'}
    
    img_path = payload['item']['album']['images'][-1]['url']
    name = payload['item']['name']

    try:
        urllib.request.urlretrieve(img_path, './tmp/thumb.png')
        img = cv2.imread('./tmp/thumb.png')
    except Exception as e:
        print(e)
        img = np.zeros((100,100,3))

    return {'image': img, 'is_playing': payload['is_playing'], 'name':name}

def led_logic(led, lock, profile=None):
    colors = COLORS
    anim = Animations(led, lock)

    while(True):
        if not lock.locked():
            colors = COLORS
            #print('new colors detected!', type(colors))

        if colors != None:
            old_colors = anim.SoftTrans()
            anim.BetweenSongTrans(COLORS, old_colors)
            #print("returned colors:", old_colors)
        else:
            #print('reseting')
            led.reset()

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

    led = LEDSim()
    #raw_output = {'timestamp': 1593381108986, 'context': {'external_urls': {'spotify': 'https://open.spotify.com/album/7mZB5aUcjoHfDKSQtDRXrf'}, 'href': 'https://api.spotify.com/v1/albums/7mZB5aUcjoHfDKSQtDRXrf', 'type': 'album', 'uri': 'spotify:album:7mZB5aUcjoHfDKSQtDRXrf'}, 'progress_ms': 135467, 'item': {'album': {'album_type': 'single', 'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/5eWpthiGD66CLD4bQjvIBp'}, 'href': 'https://api.spotify.com/v1/artists/5eWpthiGD66CLD4bQjvIBp', 'id': '5eWpthiGD66CLD4bQjvIBp', 'name': 'Sanjay Leela Bhansali', 'type': 'artist', 'uri': 'spotify:artist:5eWpthiGD66CLD4bQjvIBp'}], 'available_markets': ['AD', 'AE', 'AR', 'AT', 'AU', 'BE', 'BG', 'BH', 'BO', 'BR', 'CA', 'CH', 'CL', 'CO', 'CR', 'CY', 'CZ', 'DE', 'DK', 'DO', 'DZ', 'EC', 'EE', 'EG', 'ES', 'FI', 'FR', 'GB', 'GR', 'GT', 'HK', 'HN', 'HU', 'ID', 'IE', 'IL', 'IN', 'IS', 'IT', 'JO', 'JP', 'KW', 'LB', 'LI', 'LT', 'LU', 'LV', 'MA', 'MC', 'MT', 'MX', 'MY', 'NI', 'NL', 'NO', 'NZ', 'OM', 'PA', 'PE', 'PH', 'PL', 'PS', 'PT', 'PY', 'QA', 'RO', 'SA', 'SE', 'SG', 'SK', 'SV', 'TH', 'TN', 'TR', 'TW', 'US', 'UY', 'VN', 'ZA'], 'external_urls': {'spotify': 'https://open.spotify.com/album/7mZB5aUcjoHfDKSQtDRXrf'}, 'href': 'https://api.spotify.com/v1/albums/7mZB5aUcjoHfDKSQtDRXrf', 'id': '7mZB5aUcjoHfDKSQtDRXrf', 'images': [{'height': 640, 'url': 'https://i.scdn.co/image/ab67616d0000b273b07fa198edaf969b74dd160a', 'width': 640}, {'height': 300, 'url': 'https://i.scdn.co/image/ab67616d00001e02b07fa198edaf969b74dd160a', 'width': 300}, {'height': 64, 'url': 'https://i.scdn.co/image/ab67616d00004851b07fa198edaf969b74dd160a', 'width': 64}], 'name': 'Padmaavat', 'release_date': '2018-01-21', 'release_date_precision': 'day', 'total_tracks': 6, 'type': 'album', 'uri': 'spotify:album:7mZB5aUcjoHfDKSQtDRXrf'}, 'artists': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/0oOet2f43PA68X5RxKobEy'}, 'href': 'https://api.spotify.com/v1/artists/0oOet2f43PA68X5RxKobEy', 'id': '0oOet2f43PA68X5RxKobEy', 'name': 'Shreya Ghoshal', 'type': 'artist', 'uri': 'spotify:artist:0oOet2f43PA68X5RxKobEy'}, {'external_urls': {'spotify': 'https://open.spotify.com/artist/5hT3CHUnrkUFsgGAvEPSQC'}, 'href': 'https://api.spotify.com/v1/artists/5hT3CHUnrkUFsgGAvEPSQC', 'id': '5hT3CHUnrkUFsgGAvEPSQC', 'name': 'Swaroop Khan', 'type': 'artist', 'uri': 'spotify:artist:5hT3CHUnrkUFsgGAvEPSQC'}], 'available_markets': ['AD', 'AE', 'AR', 'AT', 'AU', 'BE', 'BG', 'BH', 'BO', 'BR', 'CA', 'CH', 'CL', 'CO', 'CR', 'CY', 'CZ', 'DE', 'DK', 'DO', 'DZ', 'EC', 'EE', 'EG', 'ES', 'FI', 'FR', 'GB', 'GR', 'GT', 'HK', 'HN', 'HU', 'ID', 'IE', 'IL', 'IN', 'IS', 'IT', 'JO', 'JP', 'KW', 'LB', 'LI', 'LT', 'LU', 'LV', 'MA', 'MC', 'MT', 'MX', 'MY', 'NI', 'NL', 'NO', 'NZ', 'OM', 'PA', 'PE', 'PH', 'PL', 'PS', 'PT', 'PY', 'QA', 'RO', 'SA', 'SE', 'SG', 'SK', 'SV', 'TH', 'TN', 'TR', 'TW', 'US', 'UY', 'VN', 'ZA'], 'disc_number': 1, 'duration_ms': 282007, 'explicit': False, 'external_ids': {'isrc': 'INS181702062'}, 'external_urls': {'spotify': 'https://open.spotify.com/track/3vJidBra6TBhO022fk6h8b'}, 'href': 'https://api.spotify.com/v1/tracks/3vJidBra6TBhO022fk6h8b', 'id': '3vJidBra6TBhO022fk6h8b', 'is_local': False, 'name': 'Ghoomar', 'popularity': 48, 'preview_url': 'https://p.scdn.co/mp3-preview/d0977ade2eab87c72f5d68fb656095432e7ee987?cid=83da7d1d9311466d91761487808719af', 'track_number': 1, 'type': 'track', 'uri': 'spotify:track:3vJidBra6TBhO022fk6h8b'}, 'currently_playing_type': 'track', 'actions': {'disallows': {'resuming': True}}, 'is_playing': True}

    while(True):
        raw_output = sp.current_playback()
        _lock.acquire()
        out = parse_current_playing(raw_output)
        _lock.release()

        if out['is_playing'] == False:
            resume_same_song = True
            COLORS = None
            time.sleep(2)
        else:
            if launch_initially == True or raw_output == None:
                time.sleep(2)
            
            if CURRENT_PLAYING_SONG != out['name'] or resume_same_song:
                resume_same_song = False
                print('generating new colors...')
                COLORS = dom_color(out['image'])
                # launch led thread:
                if launch_initially == False:
                    print('Launching LED thread...')
                    _led_thread = threading.Thread(target=led_logic, args=(led,_lock))
                    _led_thread.start()
                    launch_initially = True
                
                CURRENT_PLAYING_SONG = out['name']