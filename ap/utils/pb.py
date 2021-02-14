def get_workspace_size(workspace):

    return workspace[0][1] - workspace[0][0]


def get_heightmap_resolution(workspace_size, heightmap_size):

    return workspace_size / heightmap_size
