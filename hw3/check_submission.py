import sys

def checkFile(filename):
    p = [None]*5
    x = [None]*5
    with open(filename, 'r') as f:
        for i,l in enumerate(f):
            if l == 'p=':
                continue
            elif l == 'x=':
                continue
            elif 1 <= i <= 5:
                p_temp = [None]*5
                for j, value in enumerate(map(float, l.split(","))):
                    p_temp[j] = value
                    if value > 1:
                        print("\n\nALLOCATION PROBABILITY LARGER THAN 1, NOT A VALID MECHANISM!!!\n\n")

                p[i - 1] = p_temp
                    
            elif 7<= i <= 11:
                x_temp = [None]*5
                for j, value in enumerate(map(float, l.split(","))):
                    x_temp[j] = value

                x[i - 7] = x_temp

    print("Is your mechanism given by the below?")
    print("p=")
    for p_temp in p:
        tempString = ""
        for value in p_temp:
            tempString = tempString + str(value) + ","
        print(tempString[:-1])
    print("x=")
    for x_temp in x:
        tempString = ""
        for value in x_temp:
            tempString = tempString + str(value) + ","
        print(tempString[:-1])

    print("i.e., is the payment for a bidder who reports type 2 and external signal 3 given by " + str(x[1][2]) + "?")
    
    print("i.e., is the allocation probability for a bidder who reports type 2 and external signal 3 given by " + str(p[1][2]) + "?\n")

    print("If so, then congratulations, your submission was correctly formatted.")

                
def main(argv):
    filename = argv[0]
    checkFile(filename)

if __name__ == "__main__":
    main(sys.argv[1:])