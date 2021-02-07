# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 10:25:37 2018

@author: Ethan
"""
import random
my_randoms = random.sample(range(10001), 10000)


aList = [3, 2, 4, 7, 1, 8, 9, 0, 6, 5]
    #count = 37
#aList = [9, 8,7 , 6, 5, 4, 3, 2, 1]
    #count = 30
#aList = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    #count = 35
def heapSort(heap):
    '''
    Assumes a broken binary heap
    Compares last parent node to children, and sifts any larger children up heap
    Once largest node at root, switches for last index of list
    removes last index from binary heap, and starts again from (new) last parent node
    repeats until root node sorted
    returns statement with number of comparisons, recursions, and swaps performed by keeping variables global
    ::param heap: a list to be sorted
    '''
    global comparisons
    global recursions
    global swaps
    comparisons = 0
    swaps = 0
    recursions = 0
    global heapSize
    heapSize = len(heap)
    
    def Switch(i, x):
        '''
        Switches the numbers contained at given indices
        ::param i: index of first number to be switched
        ::param x: index of second number to be switched
        '''
        temp = heap[i]
        heap[i] = heap[x]
        heap[x] = temp
    
    
    def maxHeapify(heap, i):
        '''
        Sets node as "larger"
        Checks node has children in heap
        If node has children, compares it to them
        If left node is larger, sets it as such
        If right node is larger than "larger," sets it as such
        If "larger" is not the index of original node, switches the values of largest child and parent
             If switch is made, calls itself recursively to compare original parent to new children
        Each comparison made increases "comparison" count
        Each switch made increases "swap" count
        Each call to function increases "recursion" count
        ::param heap: list to be sorted
        ::param i: node to compare to children
        '''
        global comparisons
        global swaps
        global recursions
        global heapSize
        recursions += 1
        largest = i
        l = i * 2 + 1
        r = i * 2 + 2
        if l <= (heapSize - 1): #if there is a left child add a comparison to count and compare to parent
            comparisons += 1          
            if heap[l] > heap[i]:
                largest = l
            if r <= (heapSize - 1): #if there is a right child add a comparison to count and compare to largest
                comparisons += 1
                if heap[r] > heap[largest]:
                    largest = r
            if largest != i: #if the largest is not the parent, switch parent with larger child
                swaps += 1
                Switch(i, largest)
                maxHeapify(heap, largest)
        return comparisons, recursions            
                 
    
    def buildMaxHeap(heap):
        '''
        Repairs broken heap
        starts at last parent node and moves to root
        sifts largest children up by calling MaxHeapify
        ::param heap: list to be sorted
        '''
        for i in range(heapSize // 2 + 1, -1, -1):
            maxHeapify(heap, i)
        
    
    buildMaxHeap(heap)                  #repair broken heap and build maxheap
    for i in range(heapSize - 1, 0, -1):#for loop counting from last index in heap and moving to 0 by 1
        Switch(0, heapSize - 1)             #Switches largest element from root node with element in last node
        heapSize = heapSize - 1             #Removes last node, containing largest element from heap
        maxHeapify(heap, 0)                 #Call maxheapify on remaining heap to re-sort, starts from root and moves down to reduce number of possible recursions that comes from starting at bottom node each call
    return (print('count = {}, recursions = {}, swaps = {}'.format(comparisons, recursions, swaps)))  #Once list is sorted, returns statement of counts
      
#heapSort(aList)
        
        
        
    
    
    
         
    
    

        
'''
step 1: define buildMaxHeap
    -Take unsorted array, turn into max heap
        -largest item in array must be at root node [0]
        -each parent node should be larger than children
            - As each element is added to array, compare to parent
            - If new element greater than parent switch them before next element added
            - If new element that is swapped with parent greater than new parent swich them 
                next element added
        -not entire array must be be sorted

step 2: Swap and remove
    -Check MaxHeap meets 2 criteria listed
    -Move parent node [0] with last element of heap
    
Step 3: def Heapify
    - Remove largest element which is in last index from heap
    - Move new root node to correct place
        - Iterate through list
            - If parent node lesser than a child, check child is greater than sibling
                - Whichever child greatest switched with parent
    
Step 4: Recursion
    - Run steps 1 to 3 until 1 node remains
    - Final node will be first element of list
    
'''   

def heapUp(heap):
    global heapSize
    heapSize = len(heap)
    global comparisons
    global swaps
    comparisons = 0
    swaps = 0
    
    def heapify(heap, heapSize):
        end = 1
        while end < heapSize:
            siftUp(heap, 0, end)
            end += 1
        
        
    def siftUp(heap, start, end):
        global comparisons
        global swaps
        child = end
        while child > start:
            comparisons += 1
            parent = child // 2 - 1 + (child % 2 > 0)
            if heap[parent] < heap[child]:
                swaps += 1
                heap[parent], heap[child] = heap[child], heap[parent]
                child = parent
            else:
                return

    
    
    heapify(heap, heapSize)
    '''
    end = heapSize - 1
    while end > 0:
        heap[end], heap[0] = heap[0], heap[end]
        end -= 1
        heapify(heap, end)       
    '''
    for i in range(heapSize - 1, -1, -1):
        heap[i], heap[0] = heap[0], heap[i]
        heapSize -= 1
        heapify(heap, i) 
    return print('comparisons: {}, swaps: {}'.format(comparisons, swaps))

#heapUp(aList)      





def leafSearch(heap):
    def leafSearch(heap, i, end):
        j = i
        while i * 2 + 2 <= end:
            if heap[j * 2 + 2] > heap[j * 2 + 1]:
                j = j * 2 + 2
            else: j = j * 2 + 1
        if j * 2 + 1 <= end:
            j = j * 2 + 1
        return j
    
    def siftDown(heap, i, end):
        j = leafSearch(heap, i, end)
        while heap[i] > heap[j]:
            j = j // 2 - 1 + (j % 2 > 0)
        x = heap[j]
        heap[j] = heap[i]
        while j > i:
            x, heap[j // 2 - 1 + (j % 2 > 0)] = heap[j // 2 - 1 + (j % 2 > 0)], x
            j = j // 2 - 1 + (j % 2 > 0)
            
    for i in range(len(heap) - 1, 0, -1):
        siftDown(heap, i, 0)
            
leafSearch(aList)