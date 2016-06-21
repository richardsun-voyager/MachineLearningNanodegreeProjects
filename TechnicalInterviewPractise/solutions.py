def question1(s,t):
    #Check the input format
    if type(s) is not str or type(t) is not str:
        print 'Input Error! Please input strings!'
        return False
    if len(t)<1:
        print 't is null!'
        return False
    #Create an anagram of t
    num = len(t)
    if num==1:#if there's only one character
        return False
    elif num>1:
        t_anagram = ''
        for i in range(num-1):#an anagram pair characters of t
            t_anagram = t[-(i+1)] + t[-(i+2)]
            if t_anagram in s:# if the pair is a substring then return true
                return True
    return False

   

def question2(a):
    #Check the input
    if type(a) is not str:
        print 'Input is not string!'
        return None
    if len(a)<2:
        print 'Input is too short'
        return None
    #Insert | into the original string,i.e., 'abc'->'|a|b|c|'
    A = '|'
    for i in range(len(a)):
        A += a[i] 
        A += '|'

    P = []#Record length of palindromic for each character as a center
    str_list = []#Record palindromic substring for each character as a center
    pal_len_max = 0
    pal_max = ''
    #Traverse each character and check palindromic substring around it
    for j in range(len(A)):
        P.append(0)
        str_list.append('')
        if j>0 and j<len(A)-1:
            steps = j if j<= len(A)-j-1 else len(A)-j-1# is j bigger than half of the length
            P[j] = 1 if j%2>0 else 0 #Suppose it is symmetric around the jth character initially
            str_list[j] = A[j] if j%2>0 else '' #record the jth character if not |
            for k in range(steps):                
                if A[j-k-1] != A[j+k+1]:#if not symmetric, break
                    #P[j] = k
                    #str_list[j] = A[j-k:j+k+1]
                    break
                if (j-k-1)%2>0:#skip inserted '|'
                    P[j] += 1
                    str_list[j] = str_list[j] + A[j-k-1]#add left character
                    str_list[j] = A[j+k+1] + str_list[j]#add right character
        if P[j]>pal_len_max:#Find the longest palindromic substring
            pal_len_max = P[j]
            pal_max = str_list[j]
    if pal_len_max<2:
        print 'No palindromic substring Exists!'
        return None
    return pal_max

def question3(G):
    #Check the input
    if G == None or type(G) is not dict:
        print 'Input is wrong!'
        return None
    if len(G) == 0:
        print 'Empty input!'
        return {}
    #Get all the vertices
    vertices = G.keys()
    vertices_num = len(vertices)
    #Use Prim Tree
    vertices_prim = []
    tree_prim = dict()
    weights_prim = 0
    #Select the first vertex randomly
    current_vertex = vertices[0]
    next_vertex = None
    vertices_prim.append(current_vertex)
    #Find the nearest neighbor during each iteration
    while len(vertices_prim)<vertices_num:
        edge_weight = float('inf')
        for vertex in vertices_prim:
            edges  = G[vertex]
            #Traverse each possible edges
            for edge in edges:
                if edge[1]<edge_weight:
                    edge_weight = edge[1]
                    current_vertex = vertex
                    next_vertex = edge[0]
        #Remove selected edge from the original adjacent list
        G[current_vertex].remove((next_vertex,edge_weight))
        G[next_vertex].remove((current_vertex,edge_weight))
        #Add edges to the growing prim tree
        if current_vertex in tree_prim:
            tree_prim[current_vertex].append((next_vertex,edge_weight))
        else:
            tree_prim[current_vertex] = [(next_vertex,edge_weight)]
        if next_vertex in tree_prim:
            tree_prim[next_vertex].append((current_vertex,edge_weight))
        else:
            tree_prim[next_vertex] = [(current_vertex,edge_weight)]  
        #Add the closest neighbor
        vertices_prim.append(next_vertex)
        weights_prim += edge_weight
    
    return tree_prim

def question4(T,r,n1,n2):
    #Check the input
    if T == None or type(r) is not \
    int or type(n1) is not int or type(n2) is not int:
        print 'Invalid input!'
        return None
    node_num = len(T)#number of nodes(including leaves)
    ancestor1_nodes = [n1]#List of ancestors of the first node, including itself
    ancestor2_nodes = [n2]#List of ancestors of the first node, including itself
    node1 = n1
    node2 = n2
    #Find ancestors of the first node
    while node1!=r:
        for i in range(node_num):
            if T[i][node1]>0:#if node1's parent is node i
                ancestor1_nodes.append(i)
                node1 = i
                break
    #Find ancestors of the second node            
    while node2!=r:
        for i in range(node_num):
            if T[i][node2]>0:#if node1's parent is node i
                ancestor2_nodes.append(i)
                node2 = i
                break
        #if this is also ancestor of the first node, return
        if node2 in ancestor1_nodes:
            return node2
    
    return ancestor1_nodes, ancestor2_nodes



#Node class
class Node(object):
    def __init__(self, data):
        self.data = data    
        self.next = None

#Create a linked list based on node class
def CreateList(num = 8):
    node = Node(0)
    node.next = Node(1)
    temp = node
    for i in range(num):
        temp = temp.next
        temp.next = Node(i+2)
    return node
        
#Find mth number from the end
def question5(ll,m):
    #Check the input
    if type(ll) is None or type(m) is not int:
        print 'Invalid input!'
        return None
    nodes = []
    #Traverse all the nodes in order, save the values in a list
    while ll!= None:
        nodes.append(ll.data)
        ll = ll.next
    #for i in range(m):
        #nodes.pop()
    #If m is larger than the length of the list
    if m>len(nodes):
        print '"m" exceeds the length of the list!'
        return None
    #mth number from the end
    return  nodes[-m:][0]