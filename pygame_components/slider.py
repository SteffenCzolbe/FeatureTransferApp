import pygame
from pygame_components.colors import *

class Slider():
    def __init__(self, screen, name, val, minval, maxval, pos, bg=DARKGREY, fg=BLACK, size=(150, 50), font_name="roboto", font_size=14):
        self.val = val  # start value
        self.maxval = maxval  # maxvalmum at slider position right
        self.minval = minval  # minvalmum at slider position left
        self.xpos = pos[0]  # x-location on screen
        self.ypos = pos[1]
        self.size = size
        self.sliderx = self.size[0] / 13
        self.slidery = self.size[1] / 3 * 2
        self.sliderlength = self.size[0] - self.sliderx*2
        self.sliderheight = self.size[1] / 15
        self.bg = bg
        self.fg = fg
        self.screen = screen
        self.surf = pygame.surface.Surface(self.size)
        self.hit = False  # the hit attribute indicates slider movement due to mouse interaction

        # text
        self.font = pygame.font.SysFont(font_name, font_size)
        self.txt_surf = self.font.render(name, 1, self.fg)
        self.txt_rect = self.txt_surf.get_rect(center=(self.size[0] / 2, self.size[1]/3))

        # Static graphics - slider background #
        self.surf.fill(self.bg)
        pygame.draw.rect(self.surf, self.fg, [0, 0, self.size[0], self.size[1]], 3) # background frame
        pygame.draw.rect(self.surf, self.fg, [self.sliderx, self.slidery, self.sliderlength, self.sliderheight], 0) # slider bar

        self.surf.blit(self.txt_surf, self.txt_rect)  # this surface never changes

        # dynamic graphics - button surface #
        self.button_surf = pygame.surface.Surface((self.sliderheight*4, self.sliderheight*4))
        self.button_surf.fill(TRANS)
        self.button_surf.set_colorkey(TRANS)
        pygame.draw.circle(self.button_surf, BLACK, (int(self.sliderheight*2), int(self.sliderheight*2)), int(self.sliderheight*2), 0) # drag control element
        pygame.draw.circle(self.button_surf, GREY, (int(self.sliderheight*2), int(self.sliderheight*2)), int(self.sliderheight*1.6), 0)

    def draw(self):
        """ Combination of static and dynamic graphics in a copy of
        the basic slide surface
        """
        # static
        surf = self.surf.copy()

        # dynamic
        pos = (self.sliderx  +int((self.val-self.minval)/(self.maxval-self.minval)*self.sliderlength), self.slidery + self.sliderheight / 2)
        self.button_rect = self.button_surf.get_rect(center=pos)
        surf.blit(self.button_surf, self.button_rect)
        self.button_rect.move_ip(self.xpos, self.ypos)  # move of button box to correct screen position

        # screen
        self.screen.blit(surf, (self.xpos, self.ypos))

    def move(self):
        """
        The dynamic part; reacts to movement of the slider button.
        """
        self.val = (pygame.mouse.get_pos()[0] - self.xpos - self.sliderx) / self.sliderlength * (self.maxval - self.minval) + self.minval
        if self.val < self.minval:
            self.val = self.minval
        if self.val > self.maxval:
            self.val = self.maxval