import pygame
import numpy as np
import scipy
import matplotlib.pyplot as plt

# Initialize Pygame mixer and sound settings
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=4, buffer=1024)

# Sound settings
sample_rate = 44100  # Sample rate in Hz
freq = 440  # Frequency of the A note in Hz

# Generate time values
t = np.linspace(0, 1, int(sample_rate), False)

# Add dictionary of notes with multipliers for each note and octave
keymap = {
    pygame.K_q: 0.5,  # A
    pygame.K_w: 0.5297,  # A# / Bb
    pygame.K_e: 0.5612,  # B
    pygame.K_r: 0.5946,  # C
    pygame.K_t: 0.6299,  # C# / Db
    pygame.K_y: 0.6674,  # D
    pygame.K_u: 0.7071,  # D# / Eb
    pygame.K_i: 0.7492,  # E
    pygame.K_o: 0.7937,  # F
    pygame.K_p: 0.8409,  # F# / Gb
    pygame.K_LEFTBRACKET: 0.8909,  # G
    pygame.K_RIGHTBRACKET: 0.9439,  # G# / Ab
    pygame.K_a: 1.0  # A
}

# List of keys
keylist = [
    pygame.K_q, pygame.K_w, pygame.K_e, pygame.K_r, pygame.K_t, pygame.K_y,
    pygame.K_u, pygame.K_i, pygame.K_o, pygame.K_p, pygame.K_LEFTBRACKET,
    pygame.K_RIGHTBRACKET, pygame.K_a
]

# Pygame window setup
screen_width = 800
screen_height = 400
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Polyphonic PySynth qwerty controls w/ visual")
#colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# List to store active keys and corresponding sounds
active_keys_sounds_map = []  # [(sound, key), ...]


# Low-pass filter function 9.11.24 this does not work, having issue with butterworth filter arguments
# look into other filters
def low_pass_filter(waveform, cutoff=1000, sample_rate=44100, resonance=1.0):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist
    '''
    fs = 1000
    t0 = np.arange(1000) / fs
    fc = 30
    w = fc / (fs / 2)
    b, a = scipy.signal.butter(5, w, 'low')
'''
    # Design the filter: scipy's butterworth filter (2nd order)
    # b, a = scipy.signal.butter(5, normal_cutoff, btype='low')
    sos = scipy.signal.butter(5, normal_cutoff, btype='low')

    # Apply the filter to the waveform
    filtered_wave = scipy.signal.sosfilt(sos, waveform)
    # filtered_waveform = scipy.signal.lfilter(b, a, waveform)
    # newwave = scipy.signal.filtfilt(b, a, waveform)
    # Amplify the filtered signal to simulate resonance
    filtered_wave *= resonance

    return filtered_wave


# Distortion by clipping the waveform 9.11.24 needs to be tested
def apply_clipping(waveform, threshold=0.2):
    # Clip the waveform at the specified threshold
    clipped_waveform = np.clip(waveform, -threshold, threshold)

    # Normalize the clipped waveform back to -1 to 1 range
    return clipped_waveform / np.max(np.abs(clipped_waveform))


# distortion by nonlinear waveshaping --tanh 9.11.24 needs to be tested
def apply_distortion(waveform, gain=5.0):
    # Apply a gain to the waveform and pass it through a tanh function
    distorted_waveform = np.tanh(gain * waveform)

    return distorted_waveform


# Function to generate sound for a given frequency
def generate_sound(frequency):
    octave = .25
    waveform = 0.5 * np.sin(2 * np.pi * octave * frequency * t)
    waveform = np.int16(waveform * 32767)  # Convert to 16-bit PCM format
    return waveform
    #return pygame.mixer.Sound(waveform)


def generate_seamless_wave(frequency, duration=1.0,
                           sample_rate=44100):  # still not really making for the smoothest transitions
    num_samples = int(sample_rate * duration)

    # Generate a time array that completes an integer number of full cycles
    t = np.linspace(0, duration, num_samples, endpoint=False)
    octave = 0.125

    # Generate waveform (sine wave)
    waveform = 0.5 * np.sin(2 * np.pi * octave * frequency * t)

    # Apply a short fade-in and fade-out to smooth transitions
    fade_length = int(0.05 * sample_rate)  # 5% of wave duration
    fade_in = np.linspace(0, 1, fade_length)
    fade_out = np.linspace(1, 0, fade_length)

    # Apply fades to waveform
    waveform[:fade_length] *= fade_in  # Fade-in
    waveform[-fade_length:] *= fade_out  # Fade-out

    waveform = np.int16(waveform * 32767)  # Convert to 16-bit PCM
    return waveform
    #return pygame.mixer.Sound(waveform)


# 9.11.24 this is not working. Issue with low_pass_filter method
def generate_filtered_wave(frequency):
    octave = .25
    waveform = 0.5 * np.sin(2 * np.pi * octave * frequency * t)

    waveform = low_pass_filter(waveform)

    waveform = np.int16(waveform * 32767)  # Convert to 16-bit PCM format
    return waveform
    #return pygame.mixer.Sound(waveform)


def generate_smooth_wave(frequency):
    octave = .125
    waveform = 0.5 * np.sin(2 * np.pi * octave * frequency * t)

    # Apply a short fade-in and fade-out to smooth transitions
    fade_length = int(0.05 * sample_rate)  # 5% of wave duration
    fade_in = np.linspace(0, 1, fade_length)
    fade_out = np.linspace(1, 0, fade_length)

    # Apply fades to waveform
    waveform[:fade_length] *= fade_in  # Fade-in
    waveform[-fade_length:] *= fade_out  # Fade-out

    waveform = np.int16(waveform * 32767)
    return waveform
    #return pygame.mixer.Sound(waveform)


def generate_distort_wave(frequency):
    octave = .125
    waveform = 0.5 * np.sin(2 * np.pi * octave * frequency * t)

    # Apply a short fade-in and fade-out to smooth transitions
    fade_length = int(0.05 * sample_rate)  # 5% of wave duration
    fade_in = np.linspace(0, 1, fade_length)
    fade_out = np.linspace(1, 0, fade_length)

    # Apply fades to waveform
    waveform[:fade_length] *= fade_in  # Fade-in
    waveform[-fade_length:] *= fade_out  # Fade-out

    # waveform = low_pass_filter(waveform, resonance=1.0, cutoff=1000)
    waveform = apply_distortion(waveform, gain=10)
    waveform = apply_clipping(waveform)

    waveform = np.int16(waveform * 32767)

    return waveform
    #return pygame.mixer.Sound(waveform)


# Function to generate trianlge wave
def generate_triangle(frequency):
    octave = .25
    waveform = 0.5 * np.sin(2 * np.pi * octave * frequency * t)
    waveform = np.cumsum(np.clip(waveform * 10, -1, 1))
    waveform = waveform / max(np.abs(waveform))

    # Apply a short fade-in and fade-out to smooth transitions
    fade_length = int(0.05 * sample_rate)  # 5% of wave duration
    fade_in = np.linspace(0, 1, fade_length)
    fade_out = np.linspace(1, 0, fade_length)
    waveform[:fade_length] *= fade_in  # Fade-in
    waveform[-fade_length:] *= fade_out  # Fade-out

    waveform = np.int16(waveform * 32767)  # Convert to 16-bit PCM format
    return waveform
    #return pygame.mixer.Sound(waveform)


# function to generate square wave
def generate_square(frequency):
    octave = .25
    waveform = 0.5 * np.sin(2 * np.pi * octave * frequency * t)
    waveform = np.clip(10 * waveform, -1, 1)

    # Apply a short fade-in and fade-out to smooth transitions
    fade_length = int(0.05 * sample_rate)  # 5% of wave duration
    fade_in = np.linspace(0, 1, fade_length)
    fade_out = np.linspace(1, 0, fade_length)
    waveform[:fade_length] *= fade_in  # Fade-in
    waveform[-fade_length:] *= fade_out  # Fade-out

    waveform = np.int16(waveform * 32767)  # Convert to 16-bit PCM format
    return waveform



def getsoundfromwave(waveform):
    sound = pygame.mixer.Sound(waveform)
    return sound


def draw_waveform(waveform):
    screen.fill(BLACK)  # Clear screen
    num_samples = len(waveform)
    x_scale = screen_width / num_samples
    y_center = screen_height / 2
    #y_scale = 100
    # Find min and max values to adjust y-scale
    min_val = np.min(waveform)
    max_val = np.max(waveform)
    print(max_val, min_val)

    # Avoid division by zero and ensure scaling covers full range
    if max_val != min_val:
        y_scale = screen.get_height() / (max_val - min_val)   # Scale for y-axis
    else:
        y_scale = screen.get_height() / 2  # Default scaling if min and max are the same

    for i in range(num_samples - 1):
        x1 = i * x_scale
        y1 = y_center - waveform[i] * y_scale
        x2 = (i + 1) * x_scale
        y2 = y_center - waveform[i + 1] * y_scale
        pygame.draw.line(screen, WHITE, (x1, y1), (x2, y2), 1)

    pygame.display.flip()

# Max number of simultaneous voices (3 in this case)
max_voices = 3

# Main loop
running = True
triangle = False
square = False
sinewave = True
distortion = False
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Key pressed
        elif event.type == pygame.KEYDOWN and event.key in keylist:
            eventkey = event.key

            # check if key already pressed
            if any(key == eventkey for _, key in active_keys_sounds_map):
                continue

            multiplier = keymap[eventkey]
            #soundwave = generate_sound(multiplier * freq) #use multiplier to adjust pitch of wave
            soundwave = generate_smooth_wave(multiplier * freq)

            #soundwave = generate_distort_wave(multiplier * freq)

            #soundwave = generate_filtered_wave(multiplier*freq)
            #soundwave = generate_seamless_wave(multiplier*freq)
            #soundwave = generate_triangle(multiplier * freq)
            #soundwave= generate_square(multiplier * freq)
            # play sound in loop

            sound = getsoundfromwave(soundwave)
            sound.play(-1)
            draw_waveform(soundwave)
            '''
            if eventkey = pygame.K_l:
                distortion = True
                soundwave = generate_distort_wave(multiplier * freq)
            '''

            # Add sound and key to active list
            active_keys_sounds_map.append((sound, eventkey))

            # If more than 3 sounds are playing, stop the first one
            if len(active_keys_sounds_map) > max_voices:
                oldest_sound, oldest_key = active_keys_sounds_map.pop(0)
                oldest_sound.stop()

        # Key released
        elif event.type == pygame.KEYUP and event.key in keylist:
            eventkey = event.key

            # Find the corresponding sound and stop it
            for i, (sound, key) in enumerate(active_keys_sounds_map):
                if key == eventkey:
                    sound.stop()
                    del active_keys_sounds_map[i]
                    break

    # Draw the waveform for the first active sound
    if active_keys_sounds_map:
        current_sound, _ = active_keys_sounds_map[-1]
        waveform = generate_distort_wave(freq)  # Regenerate waveform based on frequency

        #draw_waveform(waveform)

# Quit Pygame
pygame.quit()
