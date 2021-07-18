# Reinforcement-Learning

#Importieren der Umgebungen
import numpy as np
import pylab as plt
  
#Liste von Tuple 
points_list = [(0,1), (1,5), (5,6), (5,4), (1,2), (2,3), (2,7)]
#Ziel entspricht dem Punkt 7
goal = 7
import networkx as nx 

#Graph wird erzeugt um den idealen Weg zu finden und darzustellen
G=nx.Graph() 
G.add_edges_from(points_list) 
pos = nx.spring_layout(G) 
nx.draw_networkx_nodes(G,pos) 
nx.draw_networkx_edges(G,pos) 
nx.draw_networkx_labels(G,pos) 
plt.show()#Matrixgröße wird festgelegt
MATRIX_SIZE = 8

#Matrix wird erzeugt und alle Werte werden auf -1 gesetzt (-1 entspricht einem nicht möglichen Weg)
R = np.matrix(np.ones(shape=(MATRIX_SIZE, MATRIX_SIZE))) 
R *= -1

#Matrix wird angepasst: 0 entspricht einem möglichen Weg, 
#der aber nicht direkt zum Ziel führt (viable path) 100 entspricht einem Weg der direkt zum Ziel führt
for point in points_list: 
    print(point)
    if point[1] == goal:
        R[point] = 100 
    else:
        R[point] = 0
        
    if point[0] == goal: 
        R[point[::-1]] = 100
    else: 
        R[point[::-1]]= 0

#der Endpunkt wird ebenfalls mit dem Wert 100 gesetzt. Punkt 7 erreicht, bleib bei 7        
R[goal,goal]= 100 
R

#Q-Matrix wird erzeugt
Q = np.matrix(np.zeros([MATRIX_SIZE,MATRIX_SIZE]))
#Diskontierungsfaktor wird gesetzt
gamma = 0.8
#Startpunkt
initial_state = 1

#Kontrolle welche Punkte vom aktuellen Standort aus erreichbar sind
def available_actions(state):
    current_state_row = R[state,]
    av_act = np.where(current_state_row >= 0)[1] 
    return av_act

available_act = available_actions(initial_state)

#Zufällige Auswahl zu welchem der erreichbaren Punkte sich bewegt wird
def sample_next_action(available_actions_range): 
    next_action = int(np.random.choice(available_act,1)) 
    return next_action
action = sample_next_action(available_act)

#Aktualisierung der Q-Matrix
def update(current_state, action, gamma):
    #Die maximalen Werte von Q in der entsprechenden Zeile werden als array max_index gespeichert
    max_index = np.where(Q[action,] == np.max(Q[action,]))[1] 
    #Wenn zwei Wege den gleichen Wert in der Q-Matrix aufweisen wird zufällig ein Weg ausgewählt, sonst wird der höchste Wert genommen
    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, size = 1)) 
    else:
        max_index = int(max_index) 
    max_value = Q[action, max_index]

    Q[current_state, action] = R[current_state, action] + gamma * max_value 
    #Max Value welcher dem Wert der Q-Matrix entspricht an dem das Ziel erreicht werden kann oder schon erreicht wurde wird ausgegeben
    print('max_value', R[current_state, action] + gamma * max_value)

    if (np.max(Q) > 0): 
        return(np.sum(Q/np.max(Q)*100))
    else: 
        return (0)

update(initial_state, action, gamma)

# Trainieren
scores = []
for i in range(700):
    current_state = np.random.randint(0, int(Q.shape[0])) 
    available_act = available_actions(current_state) 
    action = sample_next_action(available_act)
    score = update(current_state,action,gamma) 
    scores.append(score)
    #Der erreichte Score wird ausgegeben
    print ('Score:', str(score))
    
print("Trained Q matrix:") 
print(Q/np.max(Q)*100)

# Testen
current_state = 0 
steps = [current_state]

while current_state != 7:
    
    next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[1]

    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, size = 1))
    else:
        next_step_index = int(next_step_index)
        
    steps.append(next_step_index) 
    current_state = next_step_index

#Der ideale Weg wird ausgegeben    
print("Most efficient path:") 
print(steps)

#Verlauf der erreichten Scores wird geplottet
plt.plot(scores) 
plt.show()
