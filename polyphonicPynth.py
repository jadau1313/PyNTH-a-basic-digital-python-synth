import pygame
import numpy as np

# Initialize Pygame mixer and sound settings
pygame.init()
pygame.mixer.init()

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
screen = pygame.display.set_mode((300, 200))
pygame.display.set_caption("Press buttons to play sounds (start with qwerty)")

# List to store active keys and corresponding sounds
active_keys_sounds_map = []  # [(sound, key), ...]


# Function to generate sound for a given frequency
def generate_sound(frequency):
    waveform = 0.5 * np.sin(2 * np.pi * frequency * t)
    waveform = np.int16(waveform * 32767)  # Convert to 16-bit PCM format
    return pygame.mixer.Sound(waveform)

#Function to generate trianlge wave
def generate_triangle(frequency):
    waveform = 0.5 * np.sin(2 * np.pi * frequency * t)
    waveform = np.cumsum(np.clip(waveform * 10, -1, 1))
    waveform = waveform / max(np.abs(waveform))
    waveform = np.int16(waveform * 32767)  # Convert to 16-bit PCM format
    return pygame.mixer.Sound(waveform)

#function to generate square wave
def generate_square(frequency):
    waveform = 0.5 * np.sin(2 * np.pi * frequency * t)
    waveform = np.clip(10*waveform, -1, 1)
    waveform = np.int16(waveform * 32767)  # Convert to 16-bit PCM format
    return pygame.mixer.Sound(waveform)



# Max number of simultaneous voices (3 in this case)
max_voices = 3

# Main loop
running = True
triangle = False
square = False
sinewave = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        #Key pressed
        elif event.type == pygame.KEYDOWN and event.key in keylist:
            eventkey = event.key

            #check if key already pressed
            if any(key == eventkey for _, key in active_keys_sounds_map):
                continue

            multiplier = keymap[eventkey]
            sound = generate_sound(multiplier * freq) #use multiplier to adjust pitch of wave

            #play sound in loop
            sound.play(-1)

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

# Quit Pygame
pygame.quit()