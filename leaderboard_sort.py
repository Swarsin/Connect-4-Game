def merge_sort(arr, wins=lambda x: x[1]):
    if len(arr) < 2:
        return arr

    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]

    merge_sort(left, wins)
    merge_sort(right, wins)

    i = 0 #index for the left array 
    j = 0 #index for the right array
    k = 0 #index for the main array
    
    while i < len(left) and j < len(right): #while there are elements in the left and right arrays
        if wins(left[i]) > wins(right[j]): #if the current element in the left array is greater than the corresponding in the right array, add it to the main array
            arr[k] = left[i]
            i += 1 
        else:
            arr[k] = right[j] #else (bigger in right array) add it to the main array
            j += 1
        k += 1

    while i < len(left): #copies remaining element from the left array into the main array
        arr[k] = left[i]
        i += 1
        k += 1

    while j < len(right): #copies any remaining elements from the right array into the main array
        arr[k] = right[j]
        j += 1
        k += 1

    return arr

# Example usage

#sorted_tuples = merge_sort(list, wins=lambda x: x[1])

# for i in sorted_tuples:
#     print(i[0])

