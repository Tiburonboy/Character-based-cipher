'''Various functions used in Character based cipher jupyterLab notebook

Last updated:  12 Feb 2021
A collection of functions used across several notebooks in this project.

Revisions:
2-10-2021: spell check. Replaced int(block_size/2) with len(a) to remove
    dependence on variable block_size, need to remove commented lines later.
2-11-2021: added the function gen_round_keys().  Changed Feistel to four rounds.

'''
import numpy as np
import pickle
import sys

# Loading some objects created in the main notebook.
with open('objs.pkl', 'rb') as f:
    block_size,PT,PT_int,key,IV,Mixer,S_box,P_table,en_letter_freq = pickle.load(f)
    
def char_xor(a,b):
    '''forward character XOR function

    Addition of arguments element-wise modulo 26.
    Called char_xor because this function replaces the
    bit wise XOR usually found in encryption operations.

    Parameters:
    a, b: array like
        The arrays to be added modulo 26.

    Returns:
    y: ndarray or scalar
        The sum of a and b modulo 26, element-wise.
    '''
    return (a + b) % 26  # add the two terms modulo 26

def inv_char_xor(a,b):
    '''inverse character XOR function

    Subtraction of arguments element-wise modulo 26.
    Called inv_char_xor because this function replaces the
    bit wise XOR usually found in encryption operations.

    Parameters:
    a, b: array like, elements must be less than 26.
        The arrays to be added modulo 26.

    Returns:
    y: array like
        The difference of a and b modulo 26, element-wise.
    '''
    r = np.ones(len(b),dtype=np.uint8)
    for i in range(len(b)):
        if a[i] >= b[i]: # do normal subtraction
            r[i] = a[i] - b[i]
        if a[i] < b[i]: # need to keep result from going negative
            r[i] = (a[i]+26) - b[i]
    return r

def RF(a,k):
    '''Round function, RF()

    Also called the Feistel function.  Implements key mixing,
    Mixer matrix multiplication, substitution (S_box) and
    permutation (P_table).

    replaced int(block_size/2) with len(a) to remove
    dependence on variable block_size

    Parameters:
    a: array like
        data sub block to be operated on, length is block_size/2
    k: array like
        encryption subkey, length is block_size/2

    Returns:
    y: array like
        Result of the Feistel function.
    '''
    # xor data with the subkey and reshape for matrix multiplication
    xor_out = np.reshape(char_xor(a,k), (-1, 1))
    # use the mixer matrix to make each element
    # of s dependent on several other elements in the block
    S_box_in = (np.asarray(Mixer @ xor_out) % 26).reshape(-1)

    # run each element of S_box_in through the S_box matrix as an index
    #S_box_out = np.ones(int(block_size/2),dtype=np.uint8)
    S_box_out = np.ones(len(a),dtype=np.uint8)
    #for i in range(int(block_size/2)):
    for i in range(len(a)):
        S_box_out[i] = S_box[i][S_box_in[i]]

    # permutate the element order of S according to P
    #f = np.ones(int(block_size/2),dtype=np.uint8)
    f = np.ones(len(a),dtype=np.uint8)
    for i in range(len(f)):
        f[i] = S_box_out[P_table[i]]
    return f

def gen_round_keys(k):
    '''generate round keys from encryption key
    
    Generates the round keys from the encryption key.
    The key is divided into left and right parts.
    Intermediate keys a0, a1, a2 and a4 are generated.
    a0 is char_xor of left and right parts.  This makes a0 dependednt
    on the whole key.  a1 is char_xor of the shifted versions of 
    left and right parts, this also makes a1 dependent on whole key.
    The key is rolled right by 1/4 the key length.  This puts parts
    of the whole key in both a2 and a3.
    Four constants are derived from pi. The round function is
    then used to generate the subkeys.

    Parameters:
    k: array like
        encryption key

    Returns:
    k0, k1, k2, k3: the four round keys
    '''
    # divide the key into left and right parts
    a0 = k[0:int(len(k)/2)]
    a1 = k[int(len(k)/2):int(len(k))]

    # generate the intermediate keys by char_xor the parts together
    a0 = char_xor(a0,a1)
    a1 = char_xor(np.roll(k[0:int(len(k)/2)],1),np.roll(a1,-1))

    # makes the temp key straddle left and right parts
    atemp = np.roll(k, int(len(k)/4)) # roll 1/4th of key length
    a2 = atemp[0:int(len(k)/2)]
    a3 = atemp[int(len(k)/2):int(len(k))]

    # constants from the digits in the number pi modulo 26
    c0 = np.array([31, 41, 59, 26, 53, 58, 97, 93], dtype=np.uint8) % 26
    c1 = np.array([23, 84, 62, 64, 33, 83, 27, 95], dtype=np.uint8) % 26
    c2 = np.array([ 2, 88, 41, 97, 16, 93, 99, 37], dtype=np.uint8) % 26
    c3 = np.array([51,  5, 82,  9, 74, 94, 45, 92], dtype=np.uint8) % 26

    # generate and return the round keys by passing the intermediate keys
    # and constants through the round function
    return RF(a0,c0), RF(a1,c1), RF(a2,c2), RF(a3,c3)

def encrypt(a,k):
    '''encrypt function with 4 rounds, one block

    Implements four rounds of the encrypt Feistel network
    on one block of plaintext.

    Parameters:
    a: array like
        one block of plaintext
    k: array like
        encryption key

    Returns:
    y: one block of ciphertext
    '''
    # generate the round keys
    k0, k1, k2, k3 = gen_round_keys(k)

    # divide data into half blocks
    left_0 = a[0:int(block_size/2)]  # left side of the block
    right_0 = a[int(block_size/2):int(block_size)]  # right side of the block

    # round 0
    right_1 = char_xor(left_0,RF(right_0,k0))
    left_1 = right_0

    #round 1
    right_2 = char_xor(left_1,RF(right_1,k1))
    left_2 = right_1

    # round 2
    right_3 = char_xor(left_2,RF(right_2,k2))
    left_3 = right_2

    #round 3
    right_4 = char_xor(left_3,RF(right_3,k3))
    left_4 = right_3    

    return np.append(left_4,right_4)

def decrypt(a,k):
    '''decrypt function with 4 rounds, one block

    Implements four rounds of the decrypt Feistel network
    on one block of ciphertext.
    
    Parameters:
    a: array like
        one block of ciphertext
    k: array like
        encryption key

    Returns:
    y: one block of plaintext
    '''
    # generate the round keys
    k0, k1, k2, k3 = gen_round_keys(k)

    # divide data into half blocks
    left_4 = a[0:int(block_size/2)]
    right_4 = a[int(block_size/2):int(block_size)]

    # round 3
    left_3 = inv_char_xor(right_4,RF(left_4,k3))
    right_3 = left_4

    # round 2
    left_2 = inv_char_xor(right_3,RF(left_3,k2))
    right_2 = left_3    

    # round 1
    left_1 = inv_char_xor(right_2,RF(left_2,k1))
    right_1 = left_2

    # round 0
    left_0 = inv_char_xor(right_1,RF(left_1,k0))
    right_0 = left_1

    return np.append(left_0,right_0)

def encrypt_ECB(pt_msg, k):
    '''Encrypt message using Electronic Code Book mode.

    Implements ECB mode encryption on message.  The message must
    have a length equal to a multiple of block size.

    Parameters:
    pt_msg: array like
        plaintext message as data type uint8 with values 0 to 25.
    k: array like
        encryption key

    Returns:
    ct: ciphertext as an uint8 array
    '''
    # split the plain text message into an array of message blocks
    msg_blocks = np.array_split(pt_msg, len(pt_msg)/block_size)

    # encrypt each block and append to the list
    ct = []  # make an empty list
    for i in msg_blocks:
        ct = np.append(ct,encrypt(i,k)).astype(np.uint8)

    return ct

def decrypt_ECB(ct_msg, k):
    '''Decrypt message using Electronic Code Book mode.

    Implements ECB mode decryption on message.

    Parameters:
    ct_msg: array like
        ciphertext message as data type uint8 with values 0 to 25.
    k: array like
        encryption key

    Returns:
    pt: ciphertext as an uint8 array
    '''
    msg_blocks = np.array_split(ct_msg, len(ct_msg)/block_size)

    pt = []
    for i in msg_blocks:
        pt = np.append(pt,decrypt(i,k)).astype(np.uint8)

    return pt

def encrypt_CBC(pt_msg,k,iv):
    '''Encrypt message using Cipher Block Chaining mode.

    Implements CBC mode encryption on message.  The message must
    have a length equal to a multiple of block size.

    add new parameter iv: initialization vector

    Parameters:
    pt_msg: array like
        plaintext message as data type uint8 with values 0 to 25.
    k: array like
        encryption key
    iv: array like
        initialization vector

    Returns:
    ct: ciphertext as an uint8 array
    '''
    pt_msg_blocks = np.array_split(pt_msg, len(pt_msg)/block_size)

    ct_msg = []
    last_ct_block = iv

    for i in pt_msg_blocks:
        # xor current block with previous block of CT
        p = char_xor(i,last_ct_block).astype(np.uint8)
        current_block_ct = encrypt(p,k).astype(np.uint8)
        last_ct_block = current_block_ct  # save for next time through the loop
        ct_msg = np.append(ct_msg,current_block_ct).astype(np.uint8)

    return ct_msg

def decrypt_CBC(ct_msg,k,iv):
    '''Decrypt message using Cipher Block Chaining mode.

    Implements CBC mode decryption on message.

    Parameters:
    ct_msg: array like
        ciphertext message as data type uint8 with values 0 to 25.
    k: array like
        encryption key
    iv: array like
        initialization vector

    Returns:
    pt: ciphertext as an uint8 array
    '''
    ct_msg_blocks = np.array_split(ct_msg, len(ct_msg)/block_size)

    pt = []
    last_ct_block = iv

    for i in ct_msg_blocks:
        p = decrypt(i,k).astype(np.uint8)
        current_block_pt = inv_char_xor(p,last_ct_block).astype(np.uint8)

        last_ct_block = i  # save for next time through the loop
        pt = np.append(pt,current_block_pt).astype(np.uint8)

    return pt

# end
