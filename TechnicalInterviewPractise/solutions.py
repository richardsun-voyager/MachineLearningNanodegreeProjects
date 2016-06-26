#Map a string into a Hashmap dictionary
def stringDict(s):
    s_dict = {}
    if s is not None:
        for l in s:
            s_dict[l] = s_dict.get(l,0) + 1            
    return s_dict
            
#Given two strings s and t, determine whether some anagram of t is a substring of s
def question1(s,t):
    #Check the input format
    if type(s) is not str or type(t) is not str:
        print 'Input Error! '
        return False
    if len(t) < 1:
        print 't is empty!'
        return False
    #Create variables
    t_len = len(t)
    s_len = len(s)
    t_dict = stringDict(t)
    s_dict = {}
    if t_len > s_len:#if t is longer
        print '"t" is longer than "s"!'
        return False
    else:
        #Fetch substring of t length
        for i in range(s_len-t_len):
            s_part = s[i:(i+t_len)]
            s_dict = stringDict(s_part)
            if s_dict == t_dict:# if the pair is a substring then return true
                return True
    return False   

#Test function question1
def test_question1():
    s='udacity'
    t='adu'
    print 'Testing Case 1:', 's = "', s, '" t = "', t,'"'
    assert question1(s, t) == True
    
    t=''
    print '###################################################'
    print 'Testing Case 2:', 's = "', s, '" t = "', t,'"'
    assert question1(s, t) == False
    
    t='yirabcdefghikjlmnopqrstuvwwwwxyssssssssytz'
    print '###################################################'
    print 'Testing Case 3:', 's = "', s, '" t = "', t,'"'
    assert question1(s, t) == False
    print 'There is a warning of the input.'

#Given a string a, find the longest palindromic substring contained in a
def question2(a):
    #Check the input
    if type(a) is not str:
        print 'Input is not string!'
        return None
    if len(a) < 2:
        return a
    
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
        if j > 0 and j < len(A)-1:
            steps = j if j<= len(A)-j-1 else len(A)-j-1# is j bigger than half of the length
            P[j] = 1 if j%2>0 else 0 #Suppose it is symmetric around the jth character initially
            str_list[j] = A[j] if j%2>0 else '' #record the jth character if not '|'
            for k in range(steps):                
                if A[j-k-1] != A[j+k+1]:#if not symmetric, break
                    #P[j] = k
                    #str_list[j] = A[j-k:j+k+1]
                    break
                if (j-k-1)%2 > 0:#skip inserted '|'
                    P[j] += 1
                    str_list[j] = str_list[j] + A[j-k-1]#add left character
                    str_list[j] = A[j+k+1] + str_list[j]#add right character
        if P[j] > pal_len_max:#Find the longest palindromic substring
            pal_len_max = P[j]
            pal_max = str_list[j]
    #if pal_len_max < 2:
        #print 'No palindromic substring Exists!'
        #return None
    return pal_max

def test_question2():
    a = 'sdffdsghfgdhsjddhgfjdllal12345678kkkkkkkk87654321slorius  \
    h3546583929867346w2fghjkl 45#652562<>!@#$%~,.;jkdtabcbatfghjjhgftab'
    print 'Testing Case 1:'
    print 'a = "',a,'"'
    assert question2(a) == '12345678kkkkkkkk87654321'
    
    a = 'sssaaassssstt6789sd'
    print '################################################################'
    print 'Testing Case 2:'
    print 'a="',a,'"'
    assert question2(a) == 'sssaaasss'
    
    a = ''
    print '################################################################'
    print 'Testing Case 3:'
    print 'a="',a,'"'
    assert question2(a) == ''


    
#Given an undirected graph G, find the minimum spanning tree within G
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

def test_question3():
    #Generate an undirected graph with given number of nodes
    import numpy as np
    np.random.seed(100)
    num = 100
    G_matrix = np.zeros([num,num])
    G = {}
    #Assign random weights to the edges
    for i in range(num-1):
        for j in range(i+1,num):
            if np.random.uniform()>0.5:#Only select half of the edges
                weight = np.random.random_integers(num)
                G_matrix[i,j] = weight
                G_matrix[j,i] = weight
                current_vertex = 'n' + str(i)
                next_vertex = 'n' + str(j)
                #Build an adjacent list
                if current_vertex in G:
                    G[current_vertex].append((next_vertex,weight))
                else:
                    G[current_vertex] = [(next_vertex,weight)]
                if next_vertex in G:
                    G[next_vertex].append((current_vertex,weight))
                else:
                    G[next_vertex] = [(current_vertex,weight)]  
    print 'Testing Case 1: a graph with ', num,' nodes'
    assert question3(G) == {'n74': [('n0', 1), ('n89', 2), ('n80', 3), ('n86', 3)], 'n75': [('n86', 3), ('n5', 1), ('n87', 1), ('n97', 1)], 'n10': [('n89', 2), ('n69', 3)], 'n77': [('n86', 2), ('n96', 4)], 'n16': [('n67', 1), ('n79', 1), ('n22', 2), ('n82', 2), ('n44', 4)], 'n71': [('n34', 3), ('n70', 3)], 'n72': [('n69', 4), ('n24', 1), ('n34', 1), ('n68', 2), ('n60', 4)], 'n73': [('n96', 2), ('n76', 3)], 'n56': [('n47', 3), ('n37', 3)], 'n54': [('n97', 3), ('n23', 4)], 'n19': [('n80', 1), ('n48', 2)], 'n78': [('n95', 3), ('n82', 5)], 'n79': [('n16', 1), ('n39', 2), ('n89', 4), ('n59', 4)], 'n50': [('n82', 3), ('n31', 3)], 'n51': [('n36', 2), ('n11', 1), ('n84', 3), ('n42', 4)], 'n38': [('n81', 3), ('n95', 3), ('n0', 4), ('n48', 4), ('n30', 5)], 'n39': [('n79', 2), ('n23', 4)], 'n11': [('n51', 1), ('n14', 4)], 'n32': [('n84', 4)], 'n30': [('n35', 2), ('n55', 4), ('n38', 5)], 'n31': [('n50', 3), ('n59', 4)], 'n66': [('n94', 4)], 'n33': [('n47', 1), ('n59', 3), ('n61', 3), ('n91', 4)], 'n34': [('n72', 1), ('n28', 1), ('n71', 3), ('n93', 4)], 'n35': [('n52', 4), ('n30', 2)], 'n36': [('n94', 3), ('n51', 2), ('n29', 4)], 'n37': [('n56', 3), ('n64', 4)], 'n49': [('n9', 3)], 'n98': [('n29', 1)], 'n99': [('n89', 2), ('n40', 4)], 'n14': [('n11', 4), ('n84', 4)], 'n92': [('n80', 2), ('n52', 3)], 'n93': [('n34', 4)], 'n90': [('n94', 4), ('n44', 1)], 'n91': [('n76', 1), ('n33', 4)], 'n96': [('n77', 4), ('n73', 2), ('n22', 4)], 'n97': [('n75', 1), ('n54', 3)], 'n94': [('n5', 1), ('n67', 2), ('n36', 3), ('n66', 4), ('n90', 4)], 'n95': [('n38', 3), ('n78', 3)], 'n45': [('n63', 4)], 'n40': [('n82', 2), ('n99', 4)], 'n18': [('n29', 1)], 'n67': [('n94', 2), ('n16', 1)], 'n48': [('n19', 2), ('n38', 4)], 'n55': [('n29', 1), ('n30', 4)], 'n63': [('n22', 3), ('n20', 3), ('n45', 4), ('n64', 4)], 'n61': [('n33', 3), ('n23', 4)], 'n60': [('n72', 4), ('n88', 2)], 'n41': [('n25', 4)], 'n52': [('n92', 3), ('n35', 4)], 'n76': [('n73', 3), ('n91', 1)], 'n42': [('n59', 3), ('n43', 3), ('n51', 4)], 'n43': [('n42', 3)], 'n44': [('n90', 1), ('n16', 4), ('n46', 4)], 'n69': [('n10', 3), ('n72', 4)], 'n68': [('n72', 2)], 'n29': [('n81', 4), ('n18', 1), ('n55', 1), ('n98', 1), ('n36', 4)], 'n28': [('n34', 1)], 'n47': [('n5', 4), ('n33', 1), ('n56', 3)], 'n64': [('n24', 2), ('n63', 4), ('n37', 4)], 'n23': [('n39', 4), ('n54', 4), ('n61', 4)], 'n22': [('n16', 2), ('n63', 3), ('n96', 4)], 'n20': [('n63', 3)], 'n25': [('n2', 2), ('n41', 4)], 'n24': [('n72', 1), ('n64', 2)], 'n84': [('n51', 3), ('n14', 4), ('n32', 4)], 'n87': [('n75', 1)], 'n86': [('n74', 3), ('n77', 2), ('n75', 3)], 'n81': [('n38', 3), ('n29', 4)], 'n80': [('n0', 2), ('n19', 1), ('n92', 2), ('n74', 3)], 'n82': [('n16', 2), ('n40', 2), ('n50', 3), ('n78', 5)], 'n70': [('n71', 3)], 'n12': [('n9', 3)], 'n89': [('n74', 2), ('n10', 2), ('n99', 2), ('n2', 4), ('n79', 4)], 'n88': [('n60', 2)], 'n9': [('n59', 3), ('n12', 3), ('n49', 3)], 'n59': [('n33', 3), ('n9', 3), ('n42', 3), ('n79', 4), ('n31', 4)], 'n46': [('n44', 4)], 'n0': [('n38', 4), ('n74', 1), ('n80', 2)], 'n2': [('n89', 4), ('n25', 2)], 'n5': [('n75', 1), ('n94', 1), ('n47', 4)]}
    
    
    print '################################################################'
    G = {'A':[('B',2),('C',10)],'B':[('A',2),('C',5),('D',10)],'C':[('A',10),('B',5),('D',3)],'D':[('B',10),('C',3)]}
    print 'Testing Case 2:'
    print 'G=',G
    assert question3(G) == {'A': [('B', 2)], 'C': [('B', 5), ('D', 3)], 'B': [('A', 2), ('C', 5)], 'D': [('C', 3)]}
    
    print '################################################################' 
    G = {}
    print 'Testing Case 3:'
    print 'G=',G
    assert question3(G) ==  {}
    print 'There is a warning of the input.'


#Find the least common ancestor between two nodes on a binary search tree
def question4(T, r, n1, n2):
    #Check the input
    if T is None or type(r) is not \
    int or type(n1) is not int or type(n2) is not int:
        print 'Invalid input!'
        return None
    if len(T) == 0:
        print 'Empty Tree!'
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

def test_question4():
    #Generate a complete binary tree with given level
    import numpy as np
    level = 10
    nodes_num = 2**level - 1
    T = np.zeros([nodes_num, nodes_num])
    for l in range(level-1):
        end_node = 2**(l+1) - 2
        start_node = 2**(l+1) - 2**l - 1
        #Find children for each father node
        for i in range(start_node, end_node+1):
            T[i,2*i+1] = 1
            T[i,2*i+2] = 1
    
    print 'Testing Case 1: r=0, n1=8, n2=9'
    print 'Part of 10 Level Binary Tree:'
    print '             0        '
    print '     1                2      '
    print '  3     4          5        6  '
    print '7  8   9  10      11   12   13   14 '
    assert question4(T, 0, 8, 9) == 1
    
    print '################################################################'
    T = [[0, 1, 1, 0, 0],[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[1, 0, 0, 0, 1],[0, 0, 0, 0, 0]]
    print 'Testing Case 2: r = 3, n1 = 0, n2 = 1'  
    print 'T=', T
    assert question4(T, 3, 0, 1) == 0
    
    print '################################################################'
    T = []
    print 'Testing Case 3:r = 0, n1 = 0, n2 = 1'  
    print 'T=', T
    assert question4(T, 0, 0, 1) == None
    print 'There is a warning of input.'

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
    print temp.data
    for i in range(num-1):        
        temp = temp.next
        temp.next = Node(i+2)
        print temp.data
    print temp.next.data
    return node
        
#Find mth number from the end
#Find mth number from the end
def question5(ll, m):
    #Check the input
    if ll is None or type(m) is not int:
        print 'Invalid input!'
        return None
    
    #Create two nodes
    nodes_num = 0
    interval = 0
    node1 = ll
    node2 = ll
    #One node is m steps in advance
    while interval < m:
        node2 = node2.next
        interval += 1
        if node2 is None:
            print '"m" exceeds the length of the list!'
            return None
    #Move forward two nodes at the same pace
    while node2 is not None:
        node2 = node2.next
        node1 = node1.next    
    return  node1.data

def test_question5():
    print 'Testing Case 1:'
    ll = CreateList(100)
    assert question5(ll, 10) == 91
    
    print '################################################################'
    print 'Testing Case 2:'
    ll = CreateList(20)
    assert question5(ll, 30) == None
    print 'Expect a warning of "m"'
    
    print '################################################################'
    print 'Testing Case 3:'
    ll = None
    assert question5(ll, 30) == None
    print 'Expect a warning of input.' 
    