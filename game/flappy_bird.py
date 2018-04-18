# -*- coding: utf-8 -*-
import pygame
import random
from itertools import cycle
from game import flappy_bird_utils

FPS = 30
SCREENWIDTH = 288
SCREENHEIGHT = 512

pygame.init()
FPSCLOCK = pygame.time.Clock()
SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
pygame.display.set_caption('Flappy Bird')

IMAGES,SOUNDS, HITMASKS = flappy_bird_utils.load()
PIPEGAPSIZE = 100
BASEY = SCREENHEIGHT * 0.79 # real height(screent height - base height)

PLAYER_WIDTH = IMAGES['player'][0].get_width()
PLAYER_HEIGHT = IMAGES['player'][0].get_height()
PIPE_WIDTH = IMAGES['pipe'][0].get_width()
PIPE_HEIGHT = IMAGES['pipe'][0].get_height()
BACKGROUND_WIDTH = IMAGES['background'].get_width()

PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])

class FlappyBirdGame:
    def __init__(self):
        self.score = self.playerIndex = self.loopIter = 0
        self.playerx = int(SCREENWIDTH * 0.2)
        self.playery = int((BASEY - PLAYER_HEIGHT) / 2.0)
        self.basex = 0
        self.baseShift = IMAGES['base'].get_width() - BACKGROUND_WIDTH
        
        newPipe1 = getRandomPipe()
        newPipe2 = getRandomPipe()
        
        self.upperPipes = [
            {'x': SCREENWIDTH, 'y': newPipe1[0]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
        ]
        self.lowerPipes = [
            {'x': SCREENWIDTH, 'y': newPipe1[1]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
        ]
        
        # player velocity, max velocity, down accleration, accleration of flap
        self.pipeVelX = -4
        self.playerVelY = 0 # player's velocity along Y, default same as playerFlapped
        self.playerMaxVelY = 10
        self.playerMinVelY = -8
        self.playerAccY = 1
        self.playerFlapAcc = -7
        self.playerFlapped = False
        
    def frame_step(self, input_actions):
        pygame.event.pump()
        
        reward = 0.1
        terminal = False
        
        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!')
        
        # input_action[0] == 1: do nothing
        # input_action[1] == 1: flap the bird
        if input_actions[1] == 1:
            #
            #if self.playery > -2 * PLAYER_HEIGHT:
            #    self.playerVelY = self.playerFlapAcc
            #    self.playerFlapped = True
            # if self.player > 0:
            self.playerFlapped = True
        
        # check for score
        playerMidPos = self.playerx + int(PLAYER_WIDTH / 2.0)
        for pipe in self.upperPipes:
            pipeMidPos = pipe['x'] + int(PIPE_WIDTH / 2.0)
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                self.score += 1
                reward = 1
        
        # playerIndex basex change
        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(PLAYER_INDEX_GEN)
        self.loopIter = (self.loopIter + 1) % 30
        self.basex = -((-self.basex + 100) % self.baseShift)
        
        # player's movement
        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped and self.playery > 0:
            self.playerVelY = self.playerFlapAcc
            self.playerFlapped = False
        self.playery += self.playerVelY
        # pos clip
        self.playery = min(self.playery, BASEY - PLAYER_HEIGHT - 1)
        self.playery = max(self.playery, 0)

        
        # move pipes to left
        for upipe, lpipe in zip(self.upperPipes, self.lowerPipes):
            upipe['x'] += self.pipeVelX
            lpipe['x'] += self.pipeVelX
            
        # add new pipe when first pipe is about to touch left of screen
        if 0 < self.upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe()
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])
            
        # remove first pipe if its out if the screen
        if self.upperPipes[0]['x'] < -PIPE_WIDTH:
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)
        
        # check if crahs here
        isCrash = checkCrash({'x': self.playerx, 'y': self.playery,
                             'index': self.playerIndex},
                            self.upperPipes, self.lowerPipes)
        
        if isCrash:
            terminal = True
            self.__init__()
            reward = -1
        
        # draw sprites
        SCREEN.blit(IMAGES['background'], (0,0))
        
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))
            
        SCREEN.blit(IMAGES['base'], (self.basex, BASEY))
        # print score so player overlaps the score
        # showScore(self.score)
        SCREEN.blit(IMAGES['player'][self.playerIndex],
                    (self.playerx, self.playery))

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        FPSCLOCK.tick(FPS)
        #print self.upperPipes[0]['y'] + PIPE_HEIGHT - int(BASEY * 0.2)
        return image_data, reward, terminal

def getRandomPipe():
    """ returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    gapYs = [20, 30, 40, 50, 60, 70, 80, 90]
    gapY = random.choice(gapYs)
    
    gapY += int(BASEY * 0.2)
    pipeX = SCREENWIDTH + 10
    
    return [
        {'x': pipeX, 'y': gapY - PIPE_HEIGHT},
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE},
    ]

def checkCrash(player, upperPipes, lowerPipes):
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()
    
    if player['y'] + player['h'] >= BASEY - 1:
        return True
    else:
        playerRect = pygame.Rect(player['x'], player['y'], player['w'], player['h'])
        
        for upipe, lpipe in zip(upperPipes, lowerPipes):
            uPipeRect = pygame.Rect(upipe['x'], upipe['y'], PIPE_WIDTH, PIPE_HEIGHT)
            lPipeRect = pygame.Rect(lpipe['x'], lpipe['y'], PIPE_WIDTH, PIPE_HEIGHT)
            
            # player and upper/lower pupe hitmasker
            pHitMask = HITMASKS['player'][pi]
            uHitMask = HITMASKS['pipe'][0]
            lHitMask = HITMASKS['pipe'][1]
            
            # if bird collided with upipe or lpipe
            ucollide = pixCollision(playerRect, uPipeRect, pHitMask, uHitMask)
            lcollide = pixCollision(playerRect, lPipeRect, pHitMask, lHitMask)
            
            if ucollide or lcollide:
                return True
        return False

def pixCollision(rect1, rect2, hitmask1, hitmask2):
    rect = rect1.clip(rect2)
    
    if rect.width == 0 and rect.height == 0:
        return False
    
    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y
    
    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False