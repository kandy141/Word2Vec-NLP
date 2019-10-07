import os
import pickle
import numpy as np


model_path = './models/'
#loss_model = 'cross_entropy'
loss_model = 'nce'

model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))
print ('Picking model file: ', model_filepath)

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'r'))

"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply

v1 = embeddings[dictionary[word_id]]

==========================================================================
"""


def get_embedding_vector(string_pair):
	w1, w2 = string_pair.split(':')
	wid1, wid2 = dictionary.get(w1, None), dictionary.get(w2, None)
	return embeddings[wid1] if wid1 else None, embeddings[wid2] if wid2 else None

def min_max_indexes(magnitude_list):
	#magnitude_list.sort()
	return magnitude_list.index(min(magnitude_list)), magnitude_list.index(max(magnitude_list))

def cosine_similarity(vec1, vec2):
	from scipy import spatial
	return 1 - spatial.distance.cosine(vec1, vec2)

def cosine_similarity_1(vec1, vec2):
	from numpy import dot
	from numpy.linalg import norm
	return dot(vec1, vec2)/(norm(vec1)*norm(vec2))



dev_file = 'word_analogy_test.txt'
output_file = 'output.txt'
row = None
with open(dev_file, 'r') as inp_file:
	rows = inp_file.readlines()

i=0

with open(output_file, 'w') as fout:
	for row in rows:
		examples, choices = row.strip().split('||')
		backup_choices = choices
		examples = [x.strip('"') for x in examples.split(',')]
		choices = [x.strip('"') for x in choices.split(',')]
		#print (examples)
		#print (choices)
		
		#finding mean x(magnitude) for all examples in a row (b-a = d-c = x) to compare against choices 
		mean_x_vector = [0]
		for example in examples:
			embdw1, embdw2 = get_embedding_vector(example)
			mean_x_vector =  np.mean([mean_x_vector, embdw1 - embdw2], axis = 0)
		#magnitude of diff vector 
		x = np.linalg.norm(mean_x_vector, ord=2)
		
		#testing diff b/w words with no relation among availbale samples
		#print np.linalg.norm((embeddings[dictionary['judge']]- embeddings[dictionary['blender']]), ord=2)
		
		choice_embeddings_similarity_factor_list = []
		for choice in choices:
			embdw1, embdw2 = get_embedding_vector(choice)
			choice_embeddings_similarity_factor_list.append(cosine_similarity(mean_x_vector, embdw1-embdw2))
		#print ('kandy: ', choice_embeddings_similarity_factor_list)
		most_ill_pair, least_ill_pair =  min_max_indexes(choice_embeddings_similarity_factor_list)
		fout.write(backup_choices.replace(',', ' ') + ' "' + choices[most_ill_pair] + '" "' + choices[least_ill_pair] + '"' + '\n')

