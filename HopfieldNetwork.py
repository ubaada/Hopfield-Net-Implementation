import numpy as np
import random

# Represents a hopfield netwok
class HopfieldNetwork:
    
    # intantiate a new network, default learning rate =1
    def __init__(self, number_of_nodes, learning_rate = 1):
        self.number_of_nodes = number_of_nodes
        self.learning_rate = learning_rate
        self.weights = np.zeros((number_of_nodes, number_of_nodes))
    
    # Randomly fire nodes until the overall output doesn't change
    # match the pattern stored in the Hopefield Net.
    def calculate_output_async(self, input):
        changed = True
        temp_output = np.copy(input)
        
        while changed:
            indices = np.random.permutation(self.number_of_nodes)
            output = np.copy(temp_output)
            for i in indices:
                sum = np.dot(self.weights[i,:], output)
                raw_output[i] = sum
                if sum >= 0:
                    output[i] = 1
                else:
                    output[i] = -1
            changed = not np.array_equal(output, temp_output)
            temp_output = np.copy(output)  
        return output
        
    def calculate_output_sync(self,input):
        temp_output = np.array(input)
        temp_output2 = np.array([])
        while True:
            output = np.dot(self.weights,temp_output)
            # apply threshhold
            output[output >= 0] = 1
            output[output < 0] = -1 
            
            if np.array_equal(output, temp_output):  # check for convergence
                # print("stable state")
                return output
            elif np.array_equal(output, temp_output2): # check for oscillation
                # print("not a stable state - oscillating")
                return output
            else:
                temp_output2 = np.copy(temp_output)
                temp_output = np.copy(output)

    # Store the patterns in the Hopfield Network
    def learn(self, input):
        # hebian learning
        I = np.identity(self.number_of_nodes) # diagnol will always be 1 if input is only 1/-1
        updates = self.learning_rate * (np.outer(input,input) - I)
        updates = updates/self.number_of_nodes
        self.weights = self.weights + updates
    
    def calculate_energy(self, input):
        I = np.identity(self.number_of_nodes) # diagnol will always be 1 if input is only 1/-1
        cross_product = np.outer(input,input) - I
        energy = -(1/2) * np.sum(self.weights * cross_product)
        return energy
    