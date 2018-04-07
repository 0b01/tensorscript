keywords = "node weights graph as break const continue crate else enum extern false fn for if let match mod move return Self self true type use where while"
print('\n'.join([ 
    '{0}_lit = {{ "{0}" }}'.format(k)
    for k in keywords.split() ]))
print "keywords = {", ' | '.join([k+"_lit" for k in keywords.split()]) , "}"