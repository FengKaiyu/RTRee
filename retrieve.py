f = open('Singapore.osm', 'r')
is_in = False
fo = open('Singapore.poi.data', 'w')
content = []
tags = set([])
is_poi = False
counter = 0
for line in f:
	line = line.strip()
	if line.startswith('<node id='):
		if not is_in:
			is_in = True
			content.append(line)
		else:
			content.clear()
			content.append(line)
	elif line.startswith('</node>'):
		if not is_in:
			print('error')
			print(content)
			print(line)
		else:
			is_in = False
			if is_poi:
				for v in content:
					fo.write('{}\n'.format(v))
				fo.write('{}\n'.format(line))
				counter += 1
				if counter % 100 == 0:
					print(counter)
			content.clear()
			is_poi = False
	else:
		if is_in:
			content.append(line)
			if line.startswith('<tag k="amenity"'):
				is_poi = True
			if line.startswith('<tag k="shop"'):
				is_poi = True
			if line.startswith('<tag k="building"'):
				is_poi = True
			if line.startswith('<tag k="tourism"'):
				is_poi = True

f.close()
fo.close()