import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
from matplotlib import cm
import matplotlib
import functiiExtra
import os
import cv2
#set seed for reproducibility
np.random.seed(100)

def initialize_population(pop_size, mean_age=45, max_age=105, xbounds = [0, 1], ybounds = [0, 1]):
    '''initialized the population for the simulation

    the population matrix for this simulation has the following columns:

    0 : unique ID
    1 : current x coordinate
    2 : current y coordinate
    3 : current heading in x direction
    4 : current heading in y direction
    5 : current speed
    6 : current state (0=healthy, 1=sick, 2=immune, 3=dead)
    7 : age
    8 : infected_since (frame the person got infected)
    9 : recovery vector (used in determining when someone recovers or dies)
    '''

    #initialize population matrix
    population = np.zeros((pop_size, 10))

    #initalize unique IDs
    population[:,0] = [x for x in range(pop_size)]

    #initialize random coordinates
    population[:,1] = np.random.uniform(low = xbounds[0] + 0.05, high = xbounds[1] - 0.05, size = (pop_size,))
    population[:,2] = np.random.uniform(low = ybounds[0] + 0.05, high = ybounds[1] - 0.05, size=(pop_size,))

    #initialize random headings -1 to 1
    population[:,3] = np.random.normal(loc = 0, scale = 1/3, size=(pop_size,))
    population[:,4] = np.random.normal(loc = 0, scale = 1/3, size=(pop_size,))

    #initialize random speeds
    population[:,5] = np.random.normal(0.01, 0.01/3)

    #initalize ages
    std_age = (max_age - mean_age) / 3
    population[:,7] = np.int32(np.random.normal(loc = mean_age, scale = std_age, size=(pop_size,)))
    population[:,7] = np.clip(population[:,7], a_min = 0,  a_max = max_age) #clip those younger than 0 years

    #build recovery_vector
    #TODO: make risks age dependent
    population[:,9] = np.random.normal(loc = 0.5, scale = 0.5 / 3, size=(pop_size,))

    return population


def update_positions(population):
    '''update positions of all people

    Uses heading and speed to update all positions for
    the next time step

    Keyword arguments
    -----------------
    population : ndarray
        the array numpy containing all the population information
    '''

    #update positions
    #x
    population[:,1] = population[:,1] + (population[:,3] * population[:,5])
    #y
    population[:,2] = population[:,2] + (population [:,4] * population[:,5])

    return population


def out_of_bounds(population, xbounds, ybounds):
    #checks which people are about to go out of bounds and corrects
    #update headings and positions where out of bounds

    #update x heading
    #determine number of elements that need to be updated
    shp = population[:,3][(population[:,1] <= xbounds[:,0]) & (population[:,3] < 0)].shape
    print(shp)
    population[:,3][(population[:,1] <= xbounds[:,0]) & (population[:,3] < 0)] = np.random.normal(loc = 0.5,   scale = 0.5/3,  size = shp)

    shp = population[:,3][(population[:,1] >= xbounds[:,1]) & (population[:,3] > 0)].shape
    population[:,3][(population[:,1] >= xbounds[:,1]) & (population[:,3] > 0)] = -np.random.normal(loc = 0.5, scale = 0.5/3, size = shp)

    #update y heading
    shp = population[:,4][(population[:,2] <= ybounds[:,0]) & (population[:,4] < 0)].shape
    population[:,4][(population[:,2] <= ybounds[:,0]) & (population[:,4] < 0)] = np.random.normal(loc = 0.5, scale = 0.5/3, size = shp)

    shp = population[:,4][(population[:,2] >= ybounds[:,1]) & (population[:,4] > 0)].shape
    population[:,4][(population[:,2] >= ybounds[:,1]) & (population[:,4] > 0)] = -np.random.normal(loc = 0.5,  scale = 0.5/3, size = shp)

    return population


def update_randoms(population, heading_update_chance=0.02,  speed_update_chance=0.02):
    '''updates random states such as heading and speed'''

    #randomly update heading
    #x
    update = np.random.random(size=(pop_size,))
    shp = update[update <= heading_update_chance].shape
    population[:,3][update <= heading_update_chance] = np.random.normal(loc = 0,  scale = 1/3, size = shp)
    #y
    update = np.random.random(size=(pop_size,))
    shp = update[update <= heading_update_chance].shape
    population[:,4][update <= heading_update_chance] = np.random.normal(loc = 0, 
                                                       scale = 1/3,
                                                       size = shp)
    #randomize speeds
    update = np.random.random(size=(pop_size,))
    shp = update[update <= heading_update_chance].shape
    population[:,5][update <= heading_update_chance] = np.random.normal(loc = 0.01, 
                                                       scale = 0.01/3,
                                                       size = shp)    
    return population


def infect(population, infection_range, infection_chance, frame):
    #find new infections
    infected_previous_step = population[population[:,6] == 1]

    new_infections = []

    #if less than half are infected, slice based on infected (to speed up computation)
    if len(infected_previous_step) < (pop_size // 2):
        for patient in infected_previous_step:
            #define infection zone for patient
            infection_zone = [patient[1] - infection_range, patient[2] - infection_range, patient[1] + infection_range, patient[2] + infection_range]

            #find healthy people surrounding infected patient
            indices = np.int32(population[:,0][(infection_zone[0] < population[:,1]) & 
                                               (population[:,1] < infection_zone[2]) &
                                               (infection_zone[1] < population [:,2]) & 
                                               (population[:,2] < infection_zone[3]) &
                                               (population[:,6] == 0)])
            for idx in indices:
                #roll die to see if healthy person will be infected
                if np.random.random() < infection_chance:
                    population[idx][6] = 1
                    population[idx][8] = frame
                    new_infections.append(idx)

    else:
        #if more than half are infected slice based in healthy people (to speed up computation)
        healthy_previous_step = population[population[:,6] == 0]
        sick_previous_step = population[population[:,6] == 1]
        for person in healthy_previous_step:
            #define infecftion range around healthy person
            infection_zone = [person[1] - infection_range, person[2] - infection_range,
                              person[1] + infection_range, person[2] + infection_range]

            if person[6] == 0: #if person is not already infected, find if infected are nearby
                #find infected nearby healthy person
                poplen = len(sick_previous_step[:,6][(infection_zone[0] < sick_previous_step[:,1]) & 
                                              (sick_previous_step[:,1] < infection_zone[2]) &
                                              (infection_zone[1] < sick_previous_step [:,2]) & 
                                              (sick_previous_step[:,2] < infection_zone[3]) &
                                              (sick_previous_step[:,6] == 1)])

                if poplen > 0:
                    if np.random.random() < (infection_chance * poplen):
                        #roll die to see if healthy person will be infected
                        population[np.int32(person[0])][6] = 1
                        population[np.int32(person[0])][8] = frame
                        new_infections.append(np.int32(person[0]))

    if len(new_infections) > 0:
        print('at timestep %i these people got sick: %s' %(frame, new_infections))

    return population


def recover_or_die(population, frame, recovery_duration, mortality_chance):
    '''see whether to recover or die

    '''

    #find sick people
    sick_people = population[population[:,6] == 1]

    #define vector of how long everyone has been sick
    illness_duration_vector = frame - sick_people[:,8]
    
    recovery_odds_vector = (illness_duration_vector - recovery_duration[0]) / np.ptp(recovery_duration)
    recovery_odds_vector = np.clip(recovery_odds_vector, a_min = 0, a_max = None)

    #update states of sick people 
    indices = sick_people[:,0][recovery_odds_vector >= sick_people[:,9]]

    cured = []
    died = []

    #decide whether to die or recover
    for idx in indices:
        if np.random.random() <= mortality_chance:
            #die
            sick_people[:,6][sick_people[:,0] == idx] = 3
            died.append(np.int32(sick_people[sick_people[:,0] == idx][:,0][0]))
        else:
            #recover (become immune)
            sick_people[:,6][sick_people[:,0] == idx] = 2
            cured.append(np.int32(sick_people[sick_people[:,0] == idx][:,0][0]))

    if len(died) > 0:
        print('at timestep %i these people died: %s' %(frame, died))
    if len(cured) > 0:
        print('at timestep %i these people recovered: %s' %(frame, cured))

    #put array back into population
    population[population[:,6] == 1] = sick_people

    return population


plot_x=[]
def update(frame, population, infection_range=0.01, infection_chance=0.03,  recovery_duration=(200, 500), mortality_chance=0.02, xbounds=[0.01, 0.99], ybounds=[0.01, 0.99], wander_range=0.05, infected_plot = []):
    #add one infection to jumpstart
    if frame == 1:
        population[0][6] = 1
        population[0][8] = 5

    #update out of bounds
    #define bounds arrays
    _xbounds = np.array([[xbounds[0] + 0.02, xbounds[1] - 0.02]] * len(population))
    _ybounds = np.array([[ybounds[0] + 0.02, ybounds[1] - 0.02]] * len(population))

    population = out_of_bounds(population, _xbounds, _ybounds)

    #update randoms
    population = update_randoms(population)

    #for dead ones: set speed and heading to 0
    population[:,3:5][population[:,6] == 3] = 0

    #update positions
    population = update_positions(population)
    
    #find new infections
    population = infect(population, infection_range, infection_chance, frame)
    infected_plot.append(len(population[population[:,6] == 1]))

    #recover and die
    population = recover_or_die(population, frame, recovery_duration, mortality_chance)

    current_inf=0
    for i in range(0,len(population)):
        if(population[i][6]==1):
            current_inf+=1
    plot_x.append(current_inf)


def translate(sensor_val, in_from, in_to, out_from, out_to):
    out_range = out_to - out_from
    in_range = in_to - in_from
    in_val = sensor_val - in_from
    val=(float(in_val)/in_range)*out_range
    out_val = out_from+val
    return out_val

def draw_image(plot, final_dimension, bound_dimension, population, pixels_width, frame):
    img = Image.new('RGB', (final_dimension,final_dimension), "white") 
    pixels = img.load()

    for pip in range(0,len(population)):
        pix_color=(0,0,0)
        if(population[pip][6]==1):
            pix_color=(255,0,0)
        elif(population[pip][6]==1):
            pix_color=(255,255,255)

        mapped_x=translate(population[pip][1],0, bound_dimension, 0, final_dimension)
        mapped_y=translate(population[pip][2],0, bound_dimension, 0, final_dimension)
        for x_add in range(0, pixels_width):
            for y_add in range(0, pixels_width):
                pixels[mapped_x+x_add,mapped_y+y_add] = pix_color
         # set the colour accordingly

    dst = Image.new('RGB', (img.width, img.height + plot.height))
    dst.paste(img, (0, 0))
    dst.paste(plot, (0, img.height))

    dst.save('render/%s.png' %frame)        # View in default viewer


if __name__ == '__main__':
    #set simulation parameters
    pop_size = 2000
    simulation_steps = 10000
    xbounds = [0, 1] 
    ybounds = [0, 1]

    population = initialize_population(pop_size)

    #RUN
    nr_frames=500
    for i in range(1,nr_frames+1):
        print("Step #" + str(i))
        update(i, population, 0.01, 0.1, (200, 500), 0.02, [0.01, 0.99], [0.01, 0.99] ,0.05,)

        figure = matplotlib.pyplot.figure(figsize=(8,3))
        plot = figure.add_subplot(111)
        plot.set_title('Nr. Oameni Infectati')
        plot.set_xlim(0, nr_frames)
        plot.set_ylim(0, pop_size + 100)
        plot.plot(plot_x)
        im = functiiExtra.fig2img (figure)
        matplotlib.pyplot.close('all')
   
        draw_image(im, 800,1,population, 3,i)

    #Create video
    image_folder = 'render'
    video_name = 'video.avi'
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, 0, 10, (width,height))
    for i in range(1,nr_frames):
        image=str(i)+'.png'
        video.write(cv2.imread(os.path.join(image_folder, image)))
    cv2.destroyAllWindows()
    video.release()

    

    




