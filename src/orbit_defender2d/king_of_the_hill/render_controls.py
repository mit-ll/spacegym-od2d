# Copyright (c) 2022, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

# Button class for human interfacing with rendered display

import pygame as pg


class Button:
    def __init__(self, surface, button_type, is_paused, is_min_latency, pos, size, color, icon_color, press_color=None):
        '''
        The init method takes the surface to draw on, the type of button to create, whether the program is paused,
        whether the current phase length (latency) is equal to the minimum, the position of the button, the size of the
        button, the background color of the button, the color of the icon within the button, and, optionally, the color
        of the icon when pressed and should define the following attributes:
         - display surface
         - appropriate type
         - position and size
         - defining rectangle
         - border widths
         - appropriate colors
         - button press latency
         - latency increment size

        It should then draw the button based on its attributes
        '''
        self._surface = surface

        if button_type == 'Play/Pause' and is_paused:
            self.type = 'Play'
        elif button_type == 'Play/Pause' and not is_paused:
            self.type = 'Pause'
        else:
            self.type = button_type

        self._x, self._y = pos
        self._size = size
        self._rect = pg.Rect(self._x, self._y, self._size, self._size)
        self._b_outline_width = 4
        self._i_outline_width = 3

        self._color = color
        self._i_color = icon_color
        self._dark_color = (color[0] - 50, color[1] - 50, color[2] - 50)
        self._light_color = (color[0] + 50, color[1] + 50, color[2] + 50)

        if press_color:
            self._press_color = press_color
        else:
            self._press_color = self._i_color

        self._press_latency = 300
        self._increment_size = 500
        self.draw_button(False, is_paused, is_min_latency)

    def draw_button(self, is_pressed=False, is_paused=True, is_min_latency=False):
        """Draw the appropriate button type and update the display only where the button is located"""
        # icon is filled with icon color when not pressed, otherwise filled with a darker color
        if is_pressed:
            icon_fill = self._dark_color
        else:
            icon_fill = self._i_color

        # draw button and lines outlining button
        pg.draw.rect(self._surface, self._color, self._rect)
        pg.draw.line(self._surface, self._light_color, (self._x, self._y), (self._x + self._size, self._y), 
                     width=self._b_outline_width)
        pg.draw.line(self._surface, self._light_color, (self._x, self._y), (self._x, self._y + self._size), 
                     width=self._b_outline_width)
        pg.draw.line(self._surface, self._dark_color, (self._x + self._size, self._y),
                     (self._x + self._size, self._y + self._size), width=self._b_outline_width)
        pg.draw.line(self._surface, self._dark_color, (self._x, self._y + self._size),
                     (self._x + self._size, self._y + self._size), width=self._b_outline_width)

        # draw triangle typically used to represent play button
        if self.type == 'Play':
            poly = ((self._x + (self._size / 4), self._y + (self._size / 4)),
                    (self._x + (self._size / 4), self._y + (3 * self._size / 4)),
                    (self._x + (3 * self._size / 4), self._y + (self._size / 2)))
            pg.draw.polygon(self._surface, icon_fill, poly)

            # if pressed, draw colored icon outline
            if is_pressed:
                pg.draw.polygon(self._surface, self._press_color, poly, width=self._i_outline_width)
        # draw two rectangles typically used to represent pause button
        elif self.type == 'Pause':
            rect1 = (self._x + (self._size / 4), self._y + (self._size / 4), 3 * self._size / 16, self._size / 2)
            rect2 = (self._x + (9 * self._size / 16), self._y + (self._size / 4), 3 * self._size / 16, self._size / 2)
            pg.draw.rect(self._surface, icon_fill, rect1)
            pg.draw.rect(self._surface, icon_fill, rect2)

            # if pressed, draw colored icon outline
            if is_pressed:
                pg.draw.rect(self._surface, self._press_color, rect1, width=self._i_outline_width)
                pg.draw.rect(self._surface, self._press_color, rect2, width=self._i_outline_width)
        # draw arrow pointing right
        elif self.type == 'Step':
            poly = ((self._x + (self._size / 2), self._y + (self._size / 2)),  # back of arrow tip
                    (self._x + (self._size / 4), self._y + (9 * self._size / 32)),
                    (self._x + (self._size / 4), self._y + (self._size / 4)),  # top left
                    (self._x + (17 * self._size / 32), self._y + (self._size / 4)),  # top right
                    (self._x + (3 * self._size / 4), self._y + (self._size / 2)),  # arrow tip
                    (self._x + (17 * self._size / 32), self._y + (3 * self._size / 4)),  # bottom right
                    (self._x + (self._size / 4), self._y + (3 * self._size / 4)),  # bottom left
                    (self._x + (self._size / 4), self._y + (23 * self._size / 32)))

            # icon is grayed out when program is not paused
            if is_paused:
                pg.draw.polygon(self._surface, icon_fill, poly)
            else:
                pg.draw.polygon(self._surface, self._dark_color, poly)

            # if pressed, draw colored icon outline
            if is_pressed and is_paused:
                pg.draw.polygon(self._surface, self._press_color, poly, width=self._i_outline_width)
        # draw arrow pointing up
        elif self.type == 'SpeedUp':
            poly = ((self._x + (self._size / 2), self._y + (self._size / 2)),
                    (self._x + (9 * self._size / 32), self._y + (3 * self._size / 4)),
                    (self._x + (self._size / 4), self._y + (3 * self._size / 4)),
                    (self._x + (self._size / 4), self._y + (15 * self._size / 32)),
                    (self._x + (self._size / 2), self._y + (self._size / 4)),
                    (self._x + (3 * self._size / 4), self._y + (15 * self._size / 32)),
                    (self._x + (3 * self._size / 4), self._y + (3 * self._size / 4)),
                    (self._x + (23 * self._size / 32), self._y + (3 * self._size / 4)))

            # icon is grayed out when latency can't be decremented any further
            if not is_min_latency:
                pg.draw.polygon(self._surface, icon_fill, poly)
            else:
                pg.draw.polygon(self._surface, self._dark_color, poly)

            # if pressed, draw colored icon outline
            if is_pressed and not is_min_latency:
                pg.draw.polygon(self._surface, self._press_color, poly, width=self._i_outline_width)
        # draw arrow pointing down
        elif self.type == 'SlowDown':
            poly = ((self._x + (self._size / 2), self._y + (self._size / 2)),
                    (self._x + (9 * self._size / 32), self._y + (self._size / 4)),
                    (self._x + (self._size / 4), self._y + (self._size / 4)),
                    (self._x + (self._size / 4), self._y + (17 * self._size / 32)),
                    (self._x + (self._size / 2), self._y + (3 * self._size / 4)),
                    (self._x + (3 * self._size / 4), self._y + (17 * self._size / 32)),
                    (self._x + (3 * self._size / 4), self._y + (self._size / 4)),
                    (self._x + (23 * self._size / 32), self._y + (self._size / 4)))

            pg.draw.polygon(self._surface, icon_fill, poly)

            # if pressed, draw colored icon outline
            if is_pressed:
                pg.draw.polygon(self._surface, self._press_color, poly, width=self._i_outline_width)
        # draw x shape typically used to represent exit button
        elif self.type == 'Quit':
            poly = ((self._x + (self._size / 4), self._y + (3 * self._size / 8)),  # top left corner
                    (self._x + (self._size / 4), self._y + (self._size / 4)),
                    (self._x + (3 * self._size / 8), self._y + (self._size / 4)),
                    (self._x + (self._size / 2), self._y + (3 * self._size / 8)),  # top middle
                    (self._x + (5 * self._size / 8), self._y + (self._size / 4)),  # top right corner
                    (self._x + (3 * self._size / 4), self._y + (self._size / 4)),
                    (self._x + (3 * self._size / 4), self._y + (3 * self._size / 8)),
                    (self._x + (5 * self._size / 8), self._y + (self._size / 2)),  # right middle
                    (self._x + (3 * self._size / 4), self._y + (5 * self._size / 8)),  # bottom right
                    (self._x + (3 * self._size / 4), self._y + (3 * self._size / 4)),
                    (self._x + (5 * self._size / 8), self._y + (3 * self._size / 4)),
                    (self._x + (self._size / 2), self._y + (5 * self._size / 8)),  # bottom middle
                    (self._x + (3 * self._size / 8), self._y + (3 * self._size / 4)),  # bottom left
                    (self._x + (self._size / 4), self._y + (3 * self._size / 4)),
                    (self._x + (self._size / 4), self._y + (5 * self._size / 8)),
                    (self._x + (3 * self._size / 8), self._y + (self._size / 2)))  # left middle

            pg.draw.polygon(self._surface, icon_fill, poly)

            # if pressed, draw colored icon outline
            if is_pressed:
                pg.draw.polygon(self._surface, self._press_color, poly, width=self._i_outline_width)

        self.update_display()

    def update_display(self):
        '''Updates pygame display only within the button's defining rectangle'''
        pg.display.update(self._rect)

    def check_mouse_pos(self, pos):
        '''Returns whether the given position lies within the button's defining rectangle'''
        if self._rect.collidepoint(pos):
            return True

        return False

    def press(self, is_paused, latency, min_latency):
        '''Determines what type of action to take based on button type when a button or key is pressed'''
        do_step = False
        do_quit = False

        # draw button press, change button type to pause, change game state to not paused, draw new button
        if self.type == 'Play':
            self.draw_button(True)
            pg.time.wait(self._press_latency)
            self.type = 'Pause'
            is_paused = False
            self.draw_button(False)
        # draw button press, change button type to play, change game state to paused, draw new button
        elif self.type == 'Pause':
            self.draw_button(True)
            pg.time.wait(self._press_latency)
            self.type = 'Play'
            is_paused = True
            self.draw_button(False)
        # draw button press, indicate advance to next phase
        elif self.type == 'Step':
            if is_paused:
                self.draw_button(is_paused, True)
                pg.time.wait(self._press_latency)
                self.draw_button(is_paused, False)
                do_step = True
        # decrement latency, ensure latency remains at or above the minimum, draw button press if latency can be
        # decremented
        elif self.type == 'SpeedUp':
            latency -= self._increment_size

            if latency <= min_latency:
                self.draw_button(is_min_latency=True)
                return is_paused, min_latency, do_step, do_quit

            self.draw_button(True)
            pg.time.wait(self._press_latency)
            self.draw_button(False)
        # draw button press, increment latency
        elif self.type == 'SlowDown':
            self.draw_button(True)
            pg.time.wait(self._press_latency)
            self.draw_button(False)
            latency += self._increment_size
        # draw button press, indicate program exit
        elif self.type == 'Quit':
            self.draw_button(True)
            pg.time.wait(self._press_latency)
            self.draw_button(False)
            do_quit = True

        return is_paused, latency, do_step, do_quit  # variables to control program flow
