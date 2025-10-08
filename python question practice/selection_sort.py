class SelectionSort():
    def __init__(self):
        self.arr_list_length_data = []
        self.arr_list = [42,21,34,56,71,23,1,0,-1,34]
    

    # def input(self):
    #     arr_len = int(input("Enter the length of the array you want : "))
    #     for i in range(arr_len - 1):
    #         ith_item = int(input(f"select the {i}th item : "))
    #         self.arr_list_length_data.append(ith_item)
    
    def sort(self):
        print("coming inside the sort method")
        n = len(self.arr_list)
        for i in range(n):
            min_element = self.arr_list[i]
            min_index = 0
            for j in range(i+1,n):
                if self.arr_list[j] < min_element:
                    min_index = j
                    min_element = self.arr_list[j]
                    self.arr_list[min_index] = self.arr_list[i]
                    self.arr_list[i] = min_element

        print(self.arr_list)
obj = SelectionSort()
obj.sort()