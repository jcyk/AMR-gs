def match(output_file, input_file):
    block = []
    blocks = []
    for line in open(input_file, encoding='utf8').readlines():
        if line.startswith('#'):
            block.append(line)
        else:
            if block:
                blocks.append(block)
            block = []

    block1 = []
    blocks1 = []
    for line in open(output_file, encoding='utf8').readlines():
        if not line.startswith('#'):
            block1.append(line)
        else:
            if block1:
                blocks1.append(block1) 
            block1 = []
    if block1:
        blocks1.append(block1)
    assert len(blocks) == len(blocks1), (len(blocks), len(blocks1))


    with open(output_file+'.pred', 'w', encoding='utf8') as fo:
        for block, block1 in zip(blocks, blocks1):
            for line in block:
                fo.write(line)
            for line in block1:
                fo.write(line)
