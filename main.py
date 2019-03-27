import pygame
from pygame_components.colors import *
from pygame_components.button import Button
from pygame_components.slider import Slider
from model.image_generator import ImageGenerator
from model.image_preprocessor import preprocess_images



# callbacks
def btn_random():
    # get new image
    generator.random()
    generator.render()
    btn_reset_to_defaults()
    return

def btn_random_own():
    # get new image
    if generator.random_own():
        generator.render()
        btn_reset_to_defaults()
    return

def btn_reset_to_defaults():
    defaults = generator.get_defaults()
    sliders[0].val = defaults["Male"]
    sliders[1].val = defaults["Smiling"]
    sliders[2].val = defaults["Black_Hair"]
    sliders[3].val = defaults["Blond_Hair"]
    sliders[4].val = defaults["No_Beard"] * -1
    sliders[5].val = defaults["Bald"]
    sliders[6].val = defaults["Eyeglasses"]
    sliders[7].val = defaults["Attractive"]
    sliders[8].val = defaults["Heavy_Makeup"]
    sliders[9].val = defaults["Brown_Hair"]
    sliders[10].val = defaults["Gray_Hair"]
    sliders[11].val = defaults["Mustache"]
    sliders[12].val = defaults["Pale_Skin"] * -1

    apply_slider_Settings()
    generator.render()
    return

def apply_slider_Settings():
    male = sliders[0].val
    smiling = sliders[1].val
    black_hair = sliders[2].val
    blond_hair = sliders[3].val
    no_beard = sliders[4].val * -1
    bald = sliders[5].val
    glasses = sliders[6].val
    attractive = sliders[7].val
    makeup = sliders[8].val
    brown_hair = sliders[9].val
    gray_hair = sliders[10].val
    mustache = sliders[11].val
    pale = sliders[12].val * -1
    generator.set_settings(attractive, bald, black_hair, blond_hair, brown_hair, glasses, gray_hair, makeup, male, mustache, no_beard, pale, smiling)
    return



# screen setup
clock = pygame.time.Clock()
pygame.init()
pygame.display.set_caption("Stylist")
screen = pygame.display.set_mode((714, 580))

# loading screen
font = pygame.font.SysFont("roboto", 44)
txt_surf = font.render("loading ...", 1, WHITE)
screen.blit(txt_surf,(0,0))
pygame.display.flip()


# register controls
buttons = [Button(screen, "Random Image", (160, 40), btn_random, bg=DARKGREY, size=(300, 60), font_size=24), 
           Button(screen, "Own Image", (160, 110), btn_random_own, bg=DARKGREY, size=(300, 60), font_size=24),
           Button(screen, "Reset", (160, 540), btn_reset_to_defaults, bg=DARKGREY, size=(300, 60), font_size=24)]
slider_size = (150, 50)
slider_pos = (10, 150)
slider_max_val = 1.0
sliders = [Slider(screen, "Gender", 0, -slider_max_val, slider_max_val, (slider_pos[0] + slider_size[0] * 0, slider_pos[1] + slider_size[1] * 0), size=slider_size),
           Slider(screen, "Smile", 0, -slider_max_val, slider_max_val, (slider_pos[0] + slider_size[0] * 0, slider_pos[1] + slider_size[1] * 1), size=slider_size),
           Slider(screen, "Black Hair", 0, -slider_max_val, slider_max_val, (slider_pos[0] + slider_size[0] * 0, slider_pos[1] + slider_size[1] * 2), size=slider_size),
           Slider(screen, "Blonde Hair", 0, -slider_max_val, slider_max_val, (slider_pos[0] + slider_size[0] * 0, slider_pos[1] + slider_size[1] * 3), size=slider_size),
           Slider(screen, "Beard", 0, -slider_max_val, slider_max_val, (slider_pos[0] + slider_size[0] * 0, slider_pos[1] + slider_size[1] * 4), size=slider_size),
           Slider(screen, "Bald", 0, -slider_max_val, slider_max_val, (slider_pos[0] + slider_size[0] * 0, slider_pos[1] + slider_size[1] * 5), size=slider_size),
           Slider(screen, "Glasses", 0, -slider_max_val, slider_max_val, (slider_pos[0] + slider_size[0] * 0, slider_pos[1] + slider_size[1] * 6), size=slider_size),
           Slider(screen, "Attractiveness", 0, -slider_max_val, slider_max_val, (slider_pos[0] + slider_size[0] * 1, slider_pos[1] + slider_size[1] * 0), size=slider_size),
           Slider(screen, "Makeup", 0, -slider_max_val, slider_max_val, (slider_pos[0] + slider_size[0] * 1, slider_pos[1] + slider_size[1] * 1), size=slider_size),
           Slider(screen, "Brown Hair", 0, -slider_max_val, slider_max_val, (slider_pos[0] + slider_size[0] * 1, slider_pos[1] + slider_size[1] * 2), size=slider_size),
           Slider(screen, "Gray Hair", 0, -slider_max_val, slider_max_val, (slider_pos[0] + slider_size[0] * 1, slider_pos[1] + slider_size[1] * 3), size=slider_size),
           Slider(screen, "Mustache", 0, -slider_max_val, slider_max_val, (slider_pos[0] + slider_size[0] * 1, slider_pos[1] + slider_size[1] * 4), size=slider_size),
           Slider(screen, "Tan", 0, -slider_max_val, slider_max_val, (slider_pos[0] + slider_size[0] * 1, slider_pos[1] + slider_size[1] * 5), size=slider_size)]


# set up generator
preprocess_images()
generator = ImageGenerator()
generator.render()
btn_reset_to_defaults()

# start game
slider_changed = False
screen.fill(BLACK)
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # check collision
            pos = pygame.mouse.get_pos()
            for button in buttons:
                if button.rect.collidepoint(pos):
                    button.call_back() # click buttons
            for s in sliders:
                if s.button_rect.collidepoint(pos):
                    s.hit = True # notify sliders they are hit
                    slider_changed = True
        elif event.type == pygame.MOUSEBUTTONUP:
            for s in sliders:
                s.hit = False # notify sliders they are no longer hit
            slider_changed = False
    
    # Move slides
    for s in sliders:
        if s.hit:
            s.move()

    # draw elements
    for button in buttons:
        button.draw()
    for s in sliders:
        s.draw()
    
    # generate image
    apply_slider_Settings()
    if slider_changed:
        img = generator.render()
    else:
        img = generator.img
    raw_str = img.tobytes("raw", 'RGB')
    surface = pygame.image.fromstring(raw_str, img.size, 'RGB')
    screen.blit(surface,(320,98))

    # to foreground
    clock.tick(30)
    pygame.display.flip()