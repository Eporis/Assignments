# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 17:49:49 2018

@author: Ethan
"""

def heapify(arr, n, i):
    count = 0
    largest = i  
    l = 2 * i + 1     
    r = 2 * i + 2     
    if l < n and arr[i] < arr[l]:
        largest = l
    if r < n and arr[largest] < arr[r]:
        largest = r
    if largest != i:
        count += 1
        arr[i],arr[largest] = arr[largest],arr[i]
        count += heapify(arr, n, largest)
    return count

def heapSort(arr):
    n = len(arr)
    count = 0
    for i in range(n, -1, -1):
        heapify(arr, n, i)  
        count += heapify(arr, i, 0)
    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i] 
        count += heapify(arr, i, 0)
    return count