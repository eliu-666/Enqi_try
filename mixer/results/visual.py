import h5py
import glob
import time
import sys, getopt
from multiprocessing import Pool  # Parallelization
from lxml import etree as xml_element_tree  # xml tree for xdmf
import numpy as np

# Switch to activate calculation and writing of contact data
writeContact = False

# xdmf number types and precisions
XDMF_TYPES = {
    "float": ["Float", "4"],
    "float64": ["Float", "8"],
    "uint": ["UInt", "4"],
    "uint32": ["UInt", "4"],
    "uint64": ["UInt", "8"],
    "int": ["Int", "4"],
    "int32": ["Int", "4"],
    "int64": ["Int", "8"],
}


def main(argv):
    """
    Process options and arguments
    """
    try:
        opts, args = getopt.getopt(argv, "m:p:t:f:i", ["cores=", "processes=", "numInnerTetras=", "nth_file=", "intraMesh="])
    except getopt.GetoptError:
        print('visual.py -m <cores> -p <processes> -t <numInnerTetras> -f <nth_file> -i')
        sys.exit(2)
    cores = 1  # default sp...
    processes = 1
    numInnerTetras = 0
    nth_file = 1
    intraMesh = 0
    for opt, arg in opts:
        if opt == '-m':
            cores = int(arg)
            print("Multiprocessing activated, Cores used: ", cores)
        elif opt == '-p':
            processes = int(arg)
            print("Processes to calculate: ", processes)
        elif opt == '-t':
            numInnerTetras = int(arg)
            print("Number of inner tetras: ", numInnerTetras)
        elif opt == '-f':
            nth_file = int(arg)
            print("Reading each n-th file: ", nth_file)
        elif opt == '-i':
            intraMesh = 1
            print("Processing intra particle mesh!")
    return cores, processes, numInnerTetras, nth_file, intraMesh


def add_attribute_item(parent, ds, centerMode):
    attr = build_item_attribute(ds)
    name = ds.name
    ds_name = name[name.rfind("/") + 1:]
    att_attribut = {'Name': ds_name, 'Center': centerMode, 'AttributeType': "Scalar"}
    Attribut = xml_element_tree.SubElement(parent, 'Attribute', att_attribut)
    DataItem = xml_element_tree.SubElement(Attribut, 'DataItem', attr)
    DataItem.text = ds.file.filename + ":" + ds.name


def add_data_item(parent, ds, centerMode):
    attr = build_item_attribute(ds)
    name = ds.name
    ds_name = name[name.rfind("/") + 1:]
    att_attribut = {'Name': ds_name, 'Center': centerMode, 'AttributeType': "Scalar"}
    Attribut = xml_element_tree.SubElement(parent, 'Attribute', att_attribut)
    DataItem = xml_element_tree.SubElement(Attribut, 'DataItem', attr)
    DataItem.text = ds.file.filename + ":" + ds.name


def add_data_item_contact(parent, ds, centerMode):
    if ds.shape[0] > 0:
        attr = build_item_attribute(ds)
        name = ds.name
        ds_name = name[name.rfind("/") + 1:]
        att_attribut = {'Name': ds_name, 'Center': centerMode, 'AttributeType': "Scalar"}
        Attribut = xml_element_tree.SubElement(parent, 'Attribute', att_attribut)
        DataItem = xml_element_tree.SubElement(Attribut, 'DataItem', attr)
        DataItem.text = ds.file.filename + ":" + ds.name


def add_poly_data_item(parent, ds, element_proto, element_pool_visual, numInnerTetras):
    """
    Adding polyhedron data at center and faces to xml file
    """
    nHullFaces = int(element_proto["numHullFaces"][0])

    if (len(ds.shape)==2 and ds.shape[1]==nHullFaces): #interpolate face values to nodes
        nHullVertices = int(element_proto["numHullVertices"][0])

        element_count = int(element_proto["lastElement"][0])
        attr = build_poly_item_attribute(ds, (element_count, nHullVertices))
        name = ds.name
        ds_name = name[name.rfind("/") + 1:]
        att_attribut = {'Name': ds_name, 'Center': "Node", 'AttributeType': "Scalar"}
        Attribut = xml_element_tree.SubElement(parent, 'Attribute', att_attribut)
        DataItem = xml_element_tree.SubElement(Attribut, 'DataItem', attr)
        DataItem.text = element_pool_visual.file.filename + ":" + ds.name

        faces = np.array(element_proto["faces"])
        distanceWeights = np.array(element_pool_visual["distanceWeights"])
        ds_np = np.array(ds)

        new_data_item = np.zeros([element_count, nHullVertices])
        numFaceVertices = faces.shape[1]
        # Loop over the vertices of all faces
        for j in range(numFaceVertices):
            # Interpolate for all faces the jth vertex
            np.add.at(
                new_data_item,
                (np.arange(element_count)[:, None], faces[:, j][None, :]),
                distanceWeights[:, j][None, :] * ds_np
            )
        new_data_item_shape = (element_count, nHullVertices) ## TODO not only 3?
        element_pool_visual.create_dataset(ds_name, shape=new_data_item_shape, dtype=ds.dtype, data=new_data_item) #Remark: Shape does not change here, I think we need to set the shape in the DataItem for this to work
    elif len(ds.shape) == 2 and ds.shape[1] > 4:  # Prevent taking vectors and quaternions
        if ds.shape[1] == numInnerTetras:  # cell values e.g. inner_temperature
            add_data_item(parent, ds, "Grid")
    else:  # uniform values e.g. position, orientation
        add_data_item(parent, ds, "Cell")


def add_poly_intra_data_item(parent, ds, element_proto, element_pool_visual):
    """
    Adding intra particle data into xml tree
    """

    numTetras = len(element_proto["tetras"])

    # particle mesh data to paraview cells
    if (len(ds.shape) == 2 and ds.shape[1] == numTetras):

        element_count = int(element_proto["lastElement"][0])
        numTotalMeshCells = int(element_count * numTetras)

        # add cell center attribute
        number_type, precision = XDMF_TYPES[ds.dtype.name]
        attr = {'Dimensions': str(numTotalMeshCells), 'NumberType': number_type, 'Precision': precision, 'Format': "HDF"}
        name = ds.name
        ds_name = name[name.rfind("/") + 1:]
        att_attribut = {'Name': ds_name, 'Center': "Cell", 'AttributeType': "Scalar"}
        Attribut = xml_element_tree.SubElement(parent, 'Attribute', att_attribut)
        DataItem = xml_element_tree.SubElement(Attribut, 'DataItem', attr)
        DataItem.text = element_pool_visual.file.filename + ":" + ds.name

        # calc cell values
        new_data_item = np.ravel(np.array(ds))
        element_pool_visual.create_dataset(ds_name, shape=numTotalMeshCells, dtype=ds.dtype, data=new_data_item) #cell data


def build_item_attribute(dataset):
    data_type = dataset.dtype
    # split data_type into number_type and precision
    number_type, precision = XDMF_TYPES[data_type.name]
    rank = dataset.shape
    dims = " ".join([str(v) for v in rank])
    item_attrib = {'Dimensions': dims, 'NumberType': number_type, 'Precision': precision, 'Format': "HDF"}
    return item_attrib


def build_poly_item_attribute(dataset, shape):
    data_type = dataset.dtype
    # split data_type into number_type and precision
    number_type, precision = XDMF_TYPES[data_type.name]
    dims = " ".join([str(v) for v in shape])
    item_attrib = {'Dimensions': dims, 'NumberType': number_type, 'Precision': precision, 'Format': "HDF"}
    return item_attrib


def calc_additional_values(file_nr):
    #print("File number: ", file_nr)

    timestep_file = h5py.File(hdf5_file_list[file_nr], "r")
    #Create new hdf5 file
    timestep_visual_file = h5py.File(hdf5_file_list[file_nr].replace(".h5", ".visual.h5"), "w")

    if process_intra_particle_mesh:
        visual_mesh_file = h5py.File(hdf5_file_list[file_nr].replace(".h5", ".mesh_visual.h5"), "w")

    # auslesen welche Elemente in der entsprechenden Datei vorhanden sind
    element_list = list(timestep_file["ElementPool"].keys())
    # Schleife ueber diese Elemente
    for element in element_list:
        # contact_type entscheidet was zusaetzlich berechnet werden muss
        contact_type = list(timestep_file["Prototype/" + element + "/contactType"])
        # Inhalt decodieren
        contact_type = contact_type[0].decode()
        element_group = timestep_visual_file.create_group("/ElementPool/" + element)
        if process_intra_particle_mesh:
            element_group_mesh = visual_mesh_file.create_group("/ElementPool/" + element)
        # hier beginnt dann der Zauber mit den Koordinaten...
        if contact_type == "Polyhedron" or contact_type == "Icosphere" or contact_type == "Spherocylinder":
            generate_polyhedron_data(timestep_file, element, element_group)
            if process_intra_particle_mesh:
                generate_polyhedron_intra_data(timestep_file, element, element_group_mesh)
        elif contact_type == "Boundary":
            generate_boundary_data(timestep_file, element, element_group)
        elif contact_type == "Sphere":
            pass
        elif contact_type == "Cell":
            pass
        else:
            print(f"given contact type {contact_type} is unknown")

    # generate contact data
    if writeContact:  # "Contact" in timestep_file:
        contact_list = list(timestep_file["Contact"].keys())
        # access e.g. contact_points_test = np.array(timestep_file["Contact/zylinder/wall/contactPointA"])
        # e.g. contact_points_test = np.array(timestep_file["Contact/zylinder/zylinder/contactPointA"])
        middle_point_list = []  # holds point in between A and B
        total_contacts = 0
        #todo: loop over all data elements in contact. Write everything in the visual-file....needed to concacenate the different entries
        #write every contact point in an array / prepare mesh
        for element in contact_list: #element = a polyhedron or sphere proto
            pair_data = timestep_file["Contact"][element]
            for pair in pair_data: #pair = possible contact pair of element
                if "copyContactPointsAB" not in pair_data[pair]:
                    continue
                #contact_points_a = np.array(timestep_file["Contact/" + element + "/" + pair + "/contactPointA"])
                #contact_points_b = np.array(timestep_file["Contact/" + element + "/" + pair + "/contactPointB"])
                contact_point_ab = np.array(timestep_file["Contact/" + element + "/" + pair + "/copyContactPointsAB"])
                num_contacts = contact_point_ab.shape[0]
                total_contacts += num_contacts
                # total_contacts_double += 2 * num_contacts #point on A and B
                for i in range(0, num_contacts): #could be made more efficient
                    # dist = contact_points_a[i] - contact_points_b[i]
                    # scaled_dist = dist * scaling
                    # middle_point = 0.5 * (contact_points_a[i] + contact_points_b[i])
                    # contact_point_list.append(contact_points_a[i] + scaled_dist)
                    # contact_point_list.append(contact_points_b[i] - scaled_dist)
                    # middle_point_list.append(middle_point)
                    middle_point_list.append(contact_point_ab[i])
                    # poly_line_entry = [0] * 2
                    # poly_line_entry[0] = offset
                    # poly_line_entry[1] = offset + 1
                    # offset += 2
                    # poly_line_list.append(poly_line_entry)

        contact_group = timestep_visual_file.create_group("/Contact")
        if total_contacts > 0:
            #defines a polyvertex
            middle_point_list_shape = (total_contacts, 3)
            contact_group.create_dataset("middlePoints", shape=middle_point_list_shape, dtype=float,
                                     data=middle_point_list)

            contact_group.create_dataset("numPolyvertices", shape=(1,), dtype=np.uint64,
                                     data=total_contacts)

            generate_contact_data(timestep_file, contact_group, contact_list)

        else:  # define a "default" contact point for when there is no contact at all
            # define polyvertex
            contact_point_list_shape = (1, 3)
            dummy_data = [-1, -1, -1]
            contact_group.create_dataset("middlePoints", shape=contact_point_list_shape, dtype=float,
                                     data=dummy_data)

            polyvertex_entry = 1
            contact_group.create_dataset("numPolyvertices", shape=(1,), dtype=np.uint64,
                                     data=polyvertex_entry)

            generate_contact_data(timestep_file, contact_group, contact_list)


    # timestep_visual_file schließen
    timestep_visual_file.close()
    # Auch die files am Ende schließen
    timestep_file.close()


def generate_contact_data(timestep_file, contact_group, contact_list):
    num_contacts = contact_group["numPolyvertices"][0]
    # prepare additional data
    data_name_list = []
    for element in contact_list:
        pair_data = timestep_file["Contact"][element]
        for cpair in pair_data:
            data_list = pair_data[cpair]
            for data in data_list:
                if data == "copyContactPointsAB":
                    continue
                data_name_list.append(data)

    data_name_set = set(data_name_list)
    data_element_list = [None] * num_contacts

    # copy data to visual
    for data_name in data_name_set:
        offset = 0
        for element in contact_list:
            pair_data = timestep_file["Contact"][element]
            for cpair in pair_data:
                if "copyContactPointsAB" not in pair_data[cpair]:
                    continue
                # data_element = pair_data[cpair + "/" + data_name]
                data_array = np.array(timestep_file["Contact/" + element + "/" + cpair + "/" + data_name])
                for data in data_array:
                    data_element_list[offset] = data
                    offset += 1
        data_shape = data_array.shape

        if data_shape[0]:
            if data_array.ndim == 1:
                new_data_shape = (num_contacts, )
                contact_group.create_dataset(data_name, shape=new_data_shape, dtype=data_array.dtype, data=data_element_list)
            if data_array.ndim == 2:
                new_data_shape = (num_contacts, data_array.shape[1])
                contact_group.create_dataset(data_name, shape=new_data_shape, dtype=data_array.dtype, data=data_element_list)
        else: #nullptr / empty data i.e. at 0.0000
            if data_array.ndim == 1:
                dummy_data = 0
                dummy_data_shape = (1, )
                if data_name == "numGJKiterations": #not nice
                    contact_group.create_dataset(data_name, shape=dummy_data_shape, dtype=np.uint64, data=dummy_data) #int data type
                else:
                    contact_group.create_dataset(data_name, shape=dummy_data_shape, dtype=float, data=dummy_data)#float data type
            if data_array.ndim == 2:
                dummy_data = [0, 0, 0]
                dummy_data_shape = (1, data_shape[1])
                contact_group.create_dataset(data_name, shape=dummy_data_shape, dtype=float, data=dummy_data)

   # data_array = np.array(timestep_file["Contact/" + element + "/" + cpair + "/" + data])
   # for i in data_array:
   #     data_element_list[offset] = data_array[i]
   #     offset += 1

   # contact_group.create_dataset()



    # contact_pool_item = hdf5_file["Contact"][item]
    # for pair in contact_pool_item:
    #     attributes_list = contact_pool_item[pair]
    #     for attr in attributes_list:
    #         attribute = attributes_list[attr]
    #         add_data_item_contact(contact_grid, attribute, "Node")


def convert_quat_to_matrix(quat):
    """
    Die hier erzeugte Matrix dreht ein Objekt in einer Operation um eine beliebige Achse
    Nicht zu verwechseln mit den Drehmatrizen der Euler-Winkel, die iterativ ausgefuehrt werden
    :param quat: uint quaternion ( quat.normalize() )
    :return: 3x3 Matrix
    """
    matrix = np.zeros(shape=(3, 3), dtype="f8")
    # Matrix Layout (ultra)
    # 1 - 2*qy2 - 2*qz2 	2*qx*qy - 2*qz*qw 	2*qx*qz + 2*qy*qw
    # 2*qx*qy + 2*qz*qw 	1 - 2*qx2 - 2*qz2 	2*qy*qz - 2*qx*qw
    # 2*qx*qz - 2*qy*qw 	2*qy*qz + 2*qx*qw 	1 - 2*qx2 - 2*qy2

    qxqw = 2 * quat.vec_part[0] * quat.w
    qxqy = 2 * quat.vec_part[0] * quat.vec_part[1]
    qxqz = 2 * quat.vec_part[0] * quat.vec_part[2]

    qyqw = 2 * quat.vec_part[1] * quat.w
    qyqz = 2 * quat.vec_part[1] * quat.vec_part[2]

    qzqw = 2 * quat.vec_part[2] * quat.w

    qx2 = 2 * quat.vec_part[0] * quat.vec_part[0]
    qy2 = 2 * quat.vec_part[1] * quat.vec_part[1]
    qz2 = 2 * quat.vec_part[2] * quat.vec_part[2]

    matrix[0][0] = 1 - qy2 - qz2
    matrix[0][1] = qxqy - qzqw
    matrix[0][2] = qxqz + qyqw

    matrix[1][0] = qxqy + qzqw
    matrix[1][1] = 1 - qx2 - qz2
    matrix[1][2] = qyqz - qxqw

    matrix[2][0] = qxqz - qyqw
    matrix[2][1] = qyqz + qxqw
    matrix[2][2] = 1 - qx2 - qy2

    return matrix


def gen_transpose_matrix3x3(matrix):
    """
    erzeugt aus der uebergebenen 3x3 Matrix die transponierte 3x3 Matrix
    :param matrix: 3x3
    :return: matrix_t 3x3
    """
    matrix_t = np.empty(shape=(3, 3), dtype="f8")

    matrix_t[0][0] = matrix[0][0]
    matrix_t[0][1] = matrix[1][0]
    matrix_t[0][2] = matrix[2][0]

    matrix_t[1][0] = matrix[0][1]
    matrix_t[1][1] = matrix[1][1]
    matrix_t[1][2] = matrix[2][1]

    matrix_t[2][0] = matrix[0][2]
    matrix_t[2][1] = matrix[1][2]
    matrix_t[2][2] = matrix[2][2]

    return matrix_t


class Quaternion:
    """
    Quaternion class
    https://mathepedia.de/Quaternionen.html
    """

    def __init__(self, w=0.0, vec_part=[]):  # x=0.0, y=0.0, z=0.0
        self.w = np.dtype("f8")  # f -> float and 8 -> for 8 Bytes
        # self.x = np.dtype("f8")
        # self.y = np.dtype("f8")
        # self.z = np.dtype("f8")
        if not len(vec_part) == 3:
            print('superGAU')
        self.w = w
        self.vec_part = np.array(vec_part, dtype=float)

    def __str__(self):
        return "w: {} x: {} y: {} z: {}".format(self.w, self.vec_part[0], self.vec_part[1], self.vec_part[2])

    def conj(self):
        return Quaternion(self.w, -self.vec_part)

    def normalize(self):
        """
        Diese Funktion ist zu 99% ueberfluessig und mathematisch nicht belegt
        :return:
        """
        s = self.w * self.w
        v = self.len_square()
        length = np.sqrt(s + v)
        if length == 0:
            return Quaternion(0, [0, 0, 0])
        return Quaternion(self.w / length,
                          [self.vec_part[0] / length, self.vec_part[1] / length, self.vec_part[2] / length])

    def len_square(self):
        len = self.vec_part[0] * self.vec_part[0] + self.vec_part[1] * self.vec_part[1] + self.vec_part[2] * \
              self.vec_part[2]
        return len

    def multiply_quat(self, q2):
        scalar = self.w * q2.w - np.dot(self.vec_part, q2.vec_part)
        vector = self.w * q2.vec_part + q2.w * self.vec_part + np.cross(self.vec_part, q2.vec_part)
        q = Quaternion(scalar, vector)
        return q

    def axis_transform(self):
        """
        Achsenwinkel-Darstellung
        :return:
        """
        angle = self.w * 0.5
        s = np.cos(angle)

        sinW = np.sin(angle)
        len_vec_part = np.sqrt(self.len_square())
        if len_vec_part <= 0:
            # print("len <= 0")
            v = [0.0, 0.0, 0.0]
        else:
            v = [self.vec_part[0] / len_vec_part, self.vec_part[1] / len_vec_part, self.vec_part[2] / len_vec_part]
            v = [v[0] * sinW, v[1] * sinW, v[2] * sinW]

        return Quaternion(s, v)

    def inverse(self):
        q_conj = self.conj()
        q_lenSq = self.len_square()

        q_inv = Quaternion(q_conj.w / q_lenSq, [q_conj.vec_part[0] / q_lenSq,
                                                q_conj.vec_part[1] / q_lenSq,
                                                q_conj.vec_part[2] / q_lenSq])
        return q_inv


def generate_polyhedron_data(file, element, element_group):
    print("Polyhedron calculation...")

    element_count = int(file["Prototype/" + element + "/lastElement"][0])
    vertices = np.array(file["Prototype/" + element + "/vertices"])
    positions = np.array(file["ElementPool/" + element + "/position"])
    scaling = np.array(file["ElementPool/" + element + "/scaling"])
    orientations = np.array(file["ElementPool/" + element + "/orientation"])
    faces = np.array(file["Prototype/" + element + "/faces"])

    num_hull_faces = int(file["Prototype/" + element + "/numHullFaces"][0])
    num_hull_vertices = int(file["Prototype/" + element + "/numHullVertices"][0])

    new_vertices_array = np.empty((element_count,num_hull_vertices, 3), dtype=np.float64)

    # Rotate, scale and translate the original prototype hull vertices for each particle
    for elem_i in range(element_count):
        orientation = Quaternion(orientations[elem_i][0], orientations[elem_i][1:])
        m = convert_quat_to_matrix(orientation)  # todo Achtung hier noch unit-quaternion erzeugen
        
        scale = np.array(scaling[elem_i])[:, np.newaxis]

        # Only take the vertices at the hull
        outer_vertices = vertices[:num_hull_vertices,:]
        
        # Apply rotation and scaling
        rotated_scaled = np.matmul(outer_vertices,m.T) * scale
        
        # Apply translation
        global_vertices = rotated_scaled + positions[elem_i] 

        new_vertices_array[elem_i,:,:] = global_vertices
        
    new_vertices_shape = (num_hull_vertices * element_count, vertices.shape[1])
    element_group.create_dataset("vertices", shape=new_vertices_shape, dtype=vertices.dtype,
                                 data=new_vertices_array)

    # Preallocate list
    new_faces_list = []

    # Create data of faces with global indices for each element
    for i in range(element_count):
        offset = i * num_hull_vertices
        face_indices = faces + offset  # shape: (num_faces, 3)
    
        # Build the interleaved structure: [3, v0, v1, v2, 3, v0, v1, v2, ...]
        face_data = np.empty((num_hull_faces, 4), dtype=int)
        face_data[:, 0] = 3  # triangle indicator
        face_data[:, 1:] = face_indices
    
        # Flatten and prepend header
        face_entry = [16, num_hull_faces]  # 16 stands for "Polyhedron" (see XDMF-Doc)
        face_entry.extend(face_data.flatten().tolist())
    
        new_faces_list.append(face_entry)
        
    new_faces_shape = (element_count, int(2 + num_hull_faces * 4))
    element_group.create_dataset("faces", shape=new_faces_shape, dtype=faces.dtype, data=new_faces_list)

    # Calculate face centers and areas currently not used
    face_center = [[] for i in range(num_hull_faces)]
    face_areas = [[] for i in range(num_hull_faces)]
    protoArea = 0
    for i in range(num_hull_faces):
        center = [0, 0, 0]
        #calc face areas for triangulatedSurfaces
        areaFace = 0
        vA = vertices[faces[i][0]]
        vB = vertices[faces[i][1]]
        vC = vertices[faces[i][2]]
        AB = vB - vA
        AC = vC - vA
        dotP = np.dot(AB, AC)
        areaFace = 0.5 * np.sqrt(np.dot(AB, AB)*np.dot(AC, AC) - dotP*dotP)
        for j in range(len(faces[i])):
            node = faces[i][j]
            center = center + vertices[node]
            #liste[node].append(i)
        center = center / 3.0
        face_center[i] = center
        face_areas[i] = areaFace
        protoArea += areaFace
    face_center_shape = (num_hull_faces, 3)
    #element_group.create_dataset("facesCenter", shape=face_center_shape, dtype=dataset_vertices.dtype, data=face_center)

    inverse_distances = [[] for i in range(num_hull_faces)]
    sum_inverse_dist = [0 for i in range(num_hull_vertices)]
    area_weight = [[] for i in range(num_hull_faces)]
    sum_area_nodes = [0 for i in range(num_hull_vertices)]
    for i in range(num_hull_faces):
        for j in range(len(faces[i])):
            node = faces[i][j]
            dist = vertices[node] - face_center[i]
            inv_dist = 1.0/(np.sqrt(dist[0] * dist[0] + dist[1] * dist[1] + dist[2] * dist[2]))
            inverse_distances[i].append(inv_dist)
            sum_inverse_dist[node] += inv_dist
            area_weight[i].append(face_areas[i])
            sum_area_nodes[node] += face_areas[i]

    for i in range(num_hull_faces):
        for j in range(len(faces[i])):
            node = faces[i][j]
            inverse_distances[i][j] /= sum_inverse_dist[node]
            area_weight[i][j] /= sum_area_nodes[node]

    inverse_distances_shape = (num_hull_faces, 3) ## TODO not only 3?
    area_weight_shape = (num_hull_faces, 3)
    #element_group.create_dataset("distanceWeights", shape=inverse_distances_shape, dtype=dataset_vertices.dtype, data=inverse_distances) #distanceWeighting
    element_group.create_dataset("distanceWeights", shape=area_weight_shape, dtype=vertices.dtype, data=area_weight) #surfaceWeighting


def generate_polyhedron_intra_data(file, element, element_group):
    print("Polyhedron intra-particle calculation...")

    element_count = int(file["Prototype/" + element + "/lastElement"][0])
    vertices = np.array(file["Prototype/" + element + "/vertices"])
    positions = np.array(file["ElementPool/" + element + "/position"])
    orientations = np.array(file["ElementPool/" + element + "/orientation"])
    tetras = np.array(file["Prototype/" + element + "/tetras"])
    faces = np.array(file["Prototype/" + element + "/faces"])
    
    numProtoMeshVertices = int(vertices.shape[0])
    numProtoTetras = int(tetras.shape[0])
    numTotalMeshVertices = int(numProtoMeshVertices * element_count)
    numTotalMeshCells = int(numProtoTetras * element_count)
    
    mesh_vertices_array = np.empty((element_count,numProtoMeshVertices, 3), dtype=np.float64)

    # Rotate and translate the original prototype vertices for each particle
    for elem_i in range(element_count):
        orientation = Quaternion(orientations[elem_i][0], orientations[elem_i][1:])
        m = convert_quat_to_matrix(orientation)  # todo Achtung hier noch unit-quaternion erzeugen
        
        # Apply rotation and scaling
        rotated_vertices = np.matmul(vertices,m.T)
        
        # Apply translation
        global_vertices = rotated_vertices + positions[elem_i] 

        mesh_vertices_array[elem_i,:,:] = global_vertices
    
    mesh_vertices_shape = (numTotalMeshVertices, vertices.shape[1])
    element_group.create_dataset("mesh_vertices", shape=mesh_vertices_shape, dtype=vertices.dtype,
                                 data=mesh_vertices_array)

    #(0, 1, 2)
    #(0, 1, 3)
    #(0, 2, 3)
    #(1, 2, 3)
    numCellFaces = 4 #  of tetrahedron faces tetras.shape[1]
    numFaceVertices = 3 # Face with 3 vertices
    subsets = [[0,1,2],[0,1,3],[0,2,3],[1,2,3]] #vertice offsets for the 4 tetrahedron faces, todo: hexahedrons

    #-------------Possibly, only call this once, because this is for a Prototype and stays constant each time step
    tetraFaces = np.zeros((numProtoTetras, numCellFaces, numFaceVertices), dtype = np.int16)
    tetraCenter = np.zeros((numProtoTetras, 3))
    tet_vertice_KO = np.zeros((4,3))
    for i in range(numProtoTetras):
        current_tetra = tetras[i]
        for s in range(4):
            tetraFaces[i,s,:] = [current_tetra[subsets[s][0]],current_tetra[subsets[s][1]],current_tetra[subsets[s][2]]]
            tet_vertice_KO[s,:] = vertices[current_tetra[s]]
        for k in range(3): #for each dimension
            tetraCenter[i][k] = (np.mean(tet_vertice_KO[:,k]))
    #-------------Possibly, only call this once, because this is for a Prototype and stays constant each time step

    entries_per_face = 1 + numFaceVertices
    entry_size = 2 + numCellFaces * entries_per_face
    total_cells = int(element_count * numProtoTetras)
    
    # Prepare offsets for all elements
    element_indices = np.arange(element_count)
    mesh_offsets = (element_indices * numProtoMeshVertices).reshape(-1, 1, 1, 1)  # shape: (E, 1, 1, 1)
    
    # Broadcast tetraFaces to all elements
    # tetraFaces shape: (T, 4, numFaceVertices)
    # Result shape: (E, T, 4, numFaceVertices)
    tetfaces_broadcasted = tetraFaces[None, :, :, :] + mesh_offsets
    
    # Flatten to (E*T, 4, numFaceVertices)
    tetfaces_flat = tetfaces_broadcasted.reshape(-1, numCellFaces, numFaceVertices)
    
    # Create output array
    new_faces_array = np.empty((total_cells, entry_size), dtype=np.int32)
    
    # Fill xdmfKey and numCellFaces
    xdmfKey = 16 #16 = Polyhedron, 6 stands for "Tetrahedron" (see XDMF-Doc), 9 for hexahedron, Note: We use polyhedron, otherwise, the topology entry in xdmf changes!
    new_faces_array[:, 0] = xdmfKey
    new_faces_array[:, 1] = numCellFaces
    
    # Fill face data
    for f in range(numCellFaces):
        face_start = 2 + f * entries_per_face
        new_faces_array[:, face_start] = numFaceVertices
        new_faces_array[:, face_start + 1: face_start + 1 + numFaceVertices] = tetfaces_flat[:, f, :]
        
    new_faces_shape = (numTotalMeshCells, int(len(new_faces_array[0])))
    element_group.create_dataset("mesh_faces", shape=new_faces_shape, dtype=faces.dtype, data=new_faces_array)

def generate_boundary_data(file, element, element_group):
    #print("Boundary Berechnungen")
    element_count = np.array(file["Prototype/" + element + "/lastElement"])[0]
    faces = np.array(file["ElementPool/" + element + "/faces/faces"])
    sizes = np.array(file["ElementPool/" + element + "/faces/sizes"])
    new_polygon_list = []
    start = 0
    end = 0
    # faces werden hier zum testen in die form gebracht um sie in paraview anzeigen zu koennen
    for i in range(0, element_count):
        # Berechnung der neuen faces
        new_polygon_list.append(3)  # 3 fix fuer polygonzug
        new_polygon_list.append(sizes[i])
        end = end + int(sizes[i])
        for j in range(start, end):
            new_polygon_list.append(faces[j])
        start = start + int(sizes[i])
    new_faces_shape = (len(new_polygon_list))
    element_group.create_dataset("polygon_faces", shape=new_faces_shape, dtype=faces.dtype, data=new_polygon_list)


def preparations(nth_file):
    """
    Preparation of hdf file name list to process.
    """
    #Create lists of hdf5 files for each processor
    hdf5_file_lists = []
    for process in range(0, processes):
        hdf5_file_list_all_times = glob.glob("P" + str(process) + "/*[!.visual].h5")
        n_times = len(hdf5_file_list_all_times)
        hdf_file_list_time_stamps = [0.0] * n_times
        #Sort list
        i_file = 0
        for file in hdf5_file_list_all_times:
            time_stamp = file.split("_")[1][:-3]
            hdf_file_list_time_stamps[i_file] = time_stamp
            i_file += 1
        hdf_file_list_time_stamps.sort(key=float)

        #Rebuild List
        prefix = "P" + str(process) + "/ld_"
        suffix = ".h5"
        new_hdf5_file_list_all_times = [None] * n_times
        for i_file in range(0, n_times):
            new_hdf5_file_list_all_times[i_file] = prefix + str(hdf_file_list_time_stamps[i_file]) + suffix

        n_postprocess = nth_file #i.e. each n-th file is postprocessed
        len_file_list = int(np.ceil(n_times/n_postprocess))
        hdf_file_list_temp = [new_hdf5_file_list_all_times[0]] * len_file_list #always store first file
        offset = n_postprocess
        for n in range(1, len_file_list):
            hdf_file_list_temp[n] = new_hdf5_file_list_all_times[offset]
            offset += n_postprocess
        hdf5_file_lists.append(hdf_file_list_temp)

        for file in hdf5_file_lists[process]:
            # oeffnen im read Modus der einzelnen hdf5 Dateien
            timestep_file = h5py.File(file, "r")
            # auslesen welche Elemente in der entsprechenden Datei vorhanden sind
            element_list = list(timestep_file["ElementPool"].keys())
            # hinzufuegen der elemete zur timestep_prototype_lists -> diese wird spaeter auf unique items ueberprueft
            timestep_prototype_lists.append(element_list)
            # Auch die files am Ende schließen
            timestep_file.close()
    return hdf5_file_lists


if __name__ == '__main__':
    # config
    user_args = main(sys.argv[1:])
    cores = user_args[0]
    processes = user_args[1]
    numInnerTetras = user_args[2]
    nth_file = user_args[3]  #each nth_hdf file is read (starting from the first file)
    process_intra_particle_mesh = user_args[4]
    # preparations
    timestep_prototype_lists = []
    hdf5_file_lists = preparations(nth_file)
    unique_prototype_list = set(item for liste in timestep_prototype_lists for item in liste)

    # Part1, creating .visual files containing additional data
    print('Creating visual HDF5 files...')
    for process in range(0, processes):
        # get hdf5 files for this process / holen der hdf5 Datei dieses Prozesses
        hdf5_file_list = hdf5_file_lists[process][:]
        # sort the list / sortiert die Liste
        hdf5_file_list.sort()
        # Liste fuer die unterschiedlichen Prototypen erzeugen -> damit Prototypen welche zu einem spaeteren Zeitpunkt in der DEM Simulation initialisiert werden ebenfalls Beruecksichtigung finden
        print('Starting calculation of additional values for P %d' %(process))
        # Create a multiprocessing Pool / erstellt einen multiprocessing Pool
        pool = Pool(processes=cores)
        # start timer / startet den Timer
        tic = time.perf_counter()
        # processing the loop by using pool / schleife ueber den pool abgearbeiten
        pool.map(calc_additional_values, range(len(hdf5_file_list)))  # process data_inputs iterable with pool
        # end timer / beendet den Timer
        toc = time.perf_counter()
        # print time elapsed / benoetigte Zeit ausgeben
        print(f"Time elapsed {toc - tic:0.4f} seconds")
        pool.close()
        pool.join()

    # Part 2 - create xdmf3 files
    # create a list containing all HDF5 Files in current folder / erstellt eine Liste mit allen HDF5 Dateien im Ordner
    new_hdf5_file_list = sum(hdf5_file_lists, [])
    proc_list = glob.glob("P*/")
    print("Creating xdmf3 - files for the following processes:")
    for p in proc_list:
        print(p)
    # sort the list / sortiert die Liste
    new_hdf5_file_list.sort()
    # Create a multiprocessing Pool / erstellt einen multiprocessing Pool
    for item in unique_prototype_list:
        # Grundgeruest fuer die xdmf Dokumente erzeugen welches dann diese Form hat:
        # <?xml version='1.0' encoding='UTF-8'?>
        # <Xdmf Version="3.0">
        #	<Domain>
        #	</Domain>
        # </Xdmf>
        root = xml_element_tree.Element("Xdmf", {"Version": "3.0"})
        domain = xml_element_tree.SubElement(root, "Domain")

        if processes == 1:
            temporal_grid = xml_element_tree.SubElement(domain, "Grid", {"GridType": "Collection", "CollectionType": "Temporal"})

        # Loop over all hdf5 files
        for new_hdf5_file in new_hdf5_file_list:
            # Get normal .h5 and visual.h5 file
            hdf5_file = h5py.File(new_hdf5_file, "r")
            hdf5_visual_file = h5py.File(new_hdf5_file.replace(".h5", ".visual.h5"), "r+")

            # in der aktuell geoeffneten hdf5 Datei verfuegbare Prototypen
            available_prototypes = list(hdf5_file["ElementPool"].keys())
            item_available = False
            # Test ob der Prototyp auch in der HDF5 Datei enthalten ist
            for pt in available_prototypes:
                if item == pt:
                    item_available = True
            if not item_available:
                break

            element_pool = hdf5_file["ElementPool"][item]  # protoname
            element_pool_visual = hdf5_visual_file["ElementPool"][item]
            element_proto = hdf5_file["Prototype"][item]
            time_dataset = hdf5_file["SimCtrl"]["simulatedTime"]
            if len(time_dataset) != 1:
                print("Something went wrong! simulatedTime dataset contains more than one entry!")
                exit()

            contact_type = hdf5_file["Prototype/" + item + "/contactType"][0].decode()

            if contact_type == "Polyhedron" or contact_type == "Icosphere" or contact_type == "Spherocylinder":
                if processes > 1: #More than one P* Folder is present
                    temporal_grid = xml_element_tree.SubElement(domain, "Grid",
                                                                {"GridType": "Collection", "CollectionType": "Temporal"})

                attr = {"Name": item, "GridType": "Uniform"}  # will gridtype be changed?
                proto_grid = xml_element_tree.SubElement(temporal_grid, "Grid", attr)

                xml_element_tree.SubElement(proto_grid, "Time", {"TimeType": "single", "Value": str(time_dataset[0])})

                # Check if there are elements in the processor, if not continue with next time step
                numberOfElements = element_pool["position"].shape[0]
                if numberOfElements == 0:
                    hdf5_file.close()
                    hdf5_visual_file.close()
                    continue

                # insert Geometry
                geo_attributes = {'GeometryType': 'XYZ'}
                geometry = xml_element_tree.SubElement(proto_grid, 'Geometry', geo_attributes)

                # insert Geometry Data Item
                geometrie = element_pool_visual["vertices"]
                geo_item_attributes = build_item_attribute(geometrie)
                data_item = xml_element_tree.SubElement(geometry, 'DataItem', geo_item_attributes)
                data_item.text = geometrie.file.filename + ":" + geometrie.name

                # insert Topology
                topo_attributes = {'TopologyType': "Mixed"}
                topology = xml_element_tree.SubElement(proto_grid, 'Topology', topo_attributes)

                # insert Topology Data Item
                topologie = element_pool_visual["faces"]
                topology_item_attributes = build_item_attribute(topologie)
                data_item = xml_element_tree.SubElement(topology, 'DataItem', topology_item_attributes)
                data_item.text = topologie.file.filename + ":" + topologie.name

                # alle uebrigen Attribute hinzufuegen
                attributes_list = [element_pool[ele_set] for ele_set in element_pool]  # All availble attributes in ele_pool

                for attribute in attributes_list:
                    add_poly_data_item(proto_grid, attribute, element_proto, element_pool_visual, numInnerTetras)

            elif contact_type == "Boundary":
                attr = {"Name": item, "GridType": "Uniform"}  # will gridtype be changed?

                if processes > 1: #More than one P* Folder is present
                    temporal_grid = xml_element_tree.SubElement(domain, "Grid", {"GridType": "Collection", "CollectionType": "Temporal"})

                attr = {"Name": item, "GridType": "Uniform"}  # will gridtype be changed?
                boundary_grid = xml_element_tree.SubElement(temporal_grid, "Grid", attr)

                xml_element_tree.SubElement(boundary_grid, "Time",{"TimeType": "single", "Value": str(time_dataset[0])})

                topo = element_pool["faces"]["faces"]
                geometrie = element_proto["vertices"]
                topo_type = contact_type

                geo_attributes = {'GeometryType': 'XYZ'}
                geometry = xml_element_tree.SubElement(boundary_grid, 'Geometry', geo_attributes)

                geo_item_attributes = build_item_attribute(geometrie)
                data_item = xml_element_tree.SubElement(geometry, 'DataItem', geo_item_attributes)
                data_item.text = geometrie.file.filename + ":" + geometrie.name

                topo_item_attributes = build_item_attribute(topo)
                # Hier werden noch nicht mehrere boundarys abgefangen... Testfall fehlt
                nodes_per_element = element_pool["faces"]["sizes"][0]
                topo_attributes = {'TopologyType': "Polygon", 'NodesPerElement': str(nodes_per_element)}
                topology = xml_element_tree.SubElement(boundary_grid, 'Topology', topo_attributes)
                data_item = xml_element_tree.SubElement(topology, 'DataItem', topo_item_attributes)
                data_item.text = topo.file.filename + ":" + topo.name
                # sofern es keine bewegte wand ist recht es die wand aus einer hdf5 datei zu laden...
                isFixed = element_proto["isFixed"]
                # Stoppen falls unbewegte Wand
                if isFixed[0] == 1 and item != "processor":
                    break

                #if uncommented, no timesteps are written
                #break

            elif contact_type == "Sphere":
                if processes > 1: #More than one P* Folder is present
                    temporal_grid = xml_element_tree.SubElement(domain, "Grid",
                                                            {"GridType": "Collection", "CollectionType": "Temporal"})
                attr = {"Name": item, "GridType": "Uniform"}  # will gridtype be changed?
                sphere_grid = xml_element_tree.SubElement(temporal_grid, "Grid", attr)

                xml_element_tree.SubElement(sphere_grid, "Time", {"TimeType": "single", "Value": str(time_dataset[0])})

                # Check if there are elements in the processor, if not continue with next time step
                numberOfElements = element_pool["position"].shape[0]
                if numberOfElements == 0:
                    hdf5_file.close()
                    hdf5_visual_file.close()
                    continue

                # insert Geometry
                geo_attributes = {'GeometryType': 'XYZ'}
                geometry = xml_element_tree.SubElement(sphere_grid, 'Geometry', geo_attributes)

                # insert Geometry Data Item
                geometrie = element_pool["position"]
                geo_item_attributes = build_item_attribute(geometrie)
                num_elements = geo_item_attributes["Dimensions"][0]
                data_item = xml_element_tree.SubElement(geometry, 'DataItem', geo_item_attributes)
                data_item.text = geometrie.file.filename + ":" + geometrie.name

                # insert Topology
                topo_attributes = {'TopologyType': 'Polyvertex', 'NumberOfElements': str(num_elements),
                                   'NodesPerElement': '1'}
                topology = xml_element_tree.SubElement(sphere_grid, 'Topology', topo_attributes)

                # alle uebrigen Attribute hinzufuegen
                attributes_list = [element_pool[ele_set] for ele_set in
                                   element_pool]  # All availble attributes in ele_pool
                for attribute in attributes_list:
                    add_data_item(sphere_grid, attribute, "Node")

            elif contact_type == "Cell":
                pass
            else:
                print(f"given contact type {contact_type} is unknown")
            hdf5_file.close()
            hdf5_visual_file.close()

        # write file
        tree = xml_element_tree.ElementTree(root)
        xml_element_tree.indent(tree, space="\t")
        tree.write(f"{item}.xdmf3", encoding="UTF-8", pretty_print=True, xml_declaration=True)
        print("done with", item)


    # Create _mesh.xdmf3
    if process_intra_particle_mesh:
        for item in unique_prototype_list:
            root = xml_element_tree.Element("Xdmf", {"Version": "3.0"})
            domain = xml_element_tree.SubElement(root, "Domain")

            if processes == 1:
                temporal_grid = xml_element_tree.SubElement(domain, "Grid", {"GridType": "Collection", "CollectionType": "Temporal"})

            # Loop over all hdf5 files
            for new_hdf5_file in new_hdf5_file_list:
                # Get normal .h5 and visual.h5 file
                hdf5_file = h5py.File(new_hdf5_file, "r")
                hdf5_visual_file = h5py.File(new_hdf5_file.replace(".h5", ".mesh_visual.h5"), "r+")

                # in der aktuell geoeffneten hdf5 Datei verfuegbare Prototypen
                available_prototypes = list(hdf5_file["ElementPool"].keys())
                item_available = False
                # Test ob der Prototyp auch in der HDF5 Datei enthalten ist
                for pt in available_prototypes:
                    if item == pt:
                        item_available = True
                if not item_available:
                    break

                element_pool = hdf5_file["ElementPool"][item]  # protoname
                element_pool_visual = hdf5_visual_file["ElementPool"][item]
                element_proto = hdf5_file["Prototype"][item]
                time_dataset = hdf5_file["SimCtrl"]["simulatedTime"]
                if len(time_dataset) != 1:
                    print("Something went wrong! simulatedTime dataset contains more than one entry!")
                    exit()

                contact_type = hdf5_file["Prototype/" + item + "/contactType"][0].decode()

                if contact_type == "Polyhedron" or contact_type == "Icosphere" or contact_type == "Spherocylinder":
                    # check if there are elements in the processor
                    numberOfElements = element_pool["position"].shape[0]
                    if numberOfElements == 0:
                        hdf5_file.close()
                        hdf5_visual_file.close()
                        continue

                    if processes > 1: #More than one P* Folder is present
                        temporal_grid = xml_element_tree.SubElement(domain, "Grid",{"GridType": "Collection", "CollectionType": "Temporal"})

                    attr = {"Name": item, "GridType": "Uniform"}  # will gridtype be changed?
                    proto_grid = xml_element_tree.SubElement(temporal_grid, "Grid", attr)

                    time = xml_element_tree.SubElement(proto_grid, "Time",
                                                       {"TimeType": "single", "Value": str(time_dataset[0])})

                    # insert Geometry
                    geo_attributes = {'GeometryType': 'XYZ'}
                    geometry = xml_element_tree.SubElement(proto_grid, 'Geometry', geo_attributes)

                    # insert Geometry Data Item
                    geometrie = element_pool_visual["mesh_vertices"]
                    geo_item_attributes = build_item_attribute(geometrie)
                    data_item = xml_element_tree.SubElement(geometry, 'DataItem', geo_item_attributes)
                    data_item.text = geometrie.file.filename + ":" + geometrie.name

                    # insert Topology
                    topo_attributes = {'TopologyType': "Mixed"}
                    topology = xml_element_tree.SubElement(proto_grid, 'Topology', topo_attributes)

                    # insert Topology Data Item
                    topologie = element_pool_visual["mesh_faces"]
                    topology_item_attributes = build_item_attribute(topologie)
                    num_elements = topology_item_attributes["Dimensions"][0]
                    data_item = xml_element_tree.SubElement(topology, 'DataItem', topology_item_attributes)
                    data_item.text = topologie.file.filename + ":" + topologie.name

                    attributes_list = [element_pool[ele_set] for ele_set in element_pool]

                    for attribute in attributes_list:
                        add_poly_intra_data_item(proto_grid, attribute, element_proto, element_pool_visual)

                hdf5_file.close()
                hdf5_visual_file.close()

            if contact_type == "Polyhedron" or contact_type == "Icosphere" or contact_type == "Spherocylinder":
                tree = xml_element_tree.ElementTree(root)
                xml_element_tree.indent(tree, space="\t")
                tree.write(f"{item}_mesh.xdmf3", encoding="UTF-8", pretty_print=True, xml_declaration=True)
                print("done with intraparticle mesh for", item)


    if writeContact:
        c_root = xml_element_tree.Element("Xdmf", {"Version": "3.0"})
        c_domain = xml_element_tree.SubElement(c_root, "Domain")
        if processes == 1:
            temporal_grid = xml_element_tree.SubElement(domain, "Grid", {"GridType": "Collection", "CollectionType": "Temporal"})
            c_temporal_grid = xml_element_tree.SubElement(c_domain, "Grid", {"GridType": "Collection", "CollectionType": "Temporal"})

        for new_hdf5_file in new_hdf5_file_list:
            hdf5_file = h5py.File(new_hdf5_file, "r")
            hdf5_visual_file = h5py.File(new_hdf5_file.replace(".h5", ".visual.h5"), "r+")
            sim_ctrl = hdf5_file["SimCtrl"]
            contact_pool_visual = hdf5_visual_file["Contact"]

            if processes > 1: #More than one P* Folder is present
                c_temporal_grid = xml_element_tree.SubElement(c_domain, "Grid",{"GridType": "Collection", "CollectionType": "Temporal"})

            attr = {"Name": "Contact", "GridType": "Uniform"}  # will gridtype be changed?
            contact_grid = xml_element_tree.SubElement(c_temporal_grid, "Grid", attr)

            # insert time
            time_dataset = sim_ctrl["simulatedTime"]
            if len(time_dataset) != 1:
                print("Something went wrong! simulatedTime dataset contains more than one entry!")
                exit()
            time = xml_element_tree.SubElement(contact_grid, "Time",
                                               {"TimeType": "single", "Value": str(time_dataset[0])})

            # insert Geometry
            geo_attributes = {'GeometryType': 'XYZ'}
            geometry = xml_element_tree.SubElement(contact_grid, 'Geometry', geo_attributes)

            # insert Geometry Data Item - polyvertex
            geometrie = contact_pool_visual["middlePoints"]
            geo_item_attributes = build_item_attribute(geometrie)
            data_item = xml_element_tree.SubElement(geometry, 'DataItem', geo_item_attributes)
            data_item.text = geometrie.file.filename + ":" + geometrie.name

            # insert Topology - Polyvertex
            num_contacts_total= geometrie.shape[0]
            topologie = contact_pool_visual["numPolyvertices"]
            topo_attributes = {'TopologyType': "Polyvertex", 'NodesPerElement': "1","NumberOfElements" :
                str(num_contacts_total)}
            topology = xml_element_tree.SubElement(contact_grid, 'Topology', topo_attributes)

            contact_pool_attributes = hdf5_visual_file["Contact"]
            for contact_attr in contact_pool_attributes:
                if contact_attr == "numPolyvertices":
                    continue
                attribute = contact_pool_attributes[contact_attr]
                add_data_item_contact(contact_grid, attribute, "Node")

            hdf5_file.close()
            hdf5_visual_file.close()

        c_tree = xml_element_tree.ElementTree(c_root)
        xml_element_tree.indent(c_tree, space="\t")
        c_tree.write("contact.xdmf", encoding="UTF-8", pretty_print=True, xml_declaration=True)
        print("done with contact.xdmf")


        # write_xmdf(item, source_files)

        # for source_file in source_files:
        #    source_file.close()
    print("end")



        # parrallel ausgeschaltet, muss aufgrund von aenderungen ueberarbeitet werden

        # pool2 = Pool(processes=cores)
        # # start timer / startet den Timer
        # tic = time.perf_counter()
        # # processing the loop by using pool / schleife ueber den pool abgearbeiten
        # pool2.map(write_xdmf, range(len(hdf5_file_list)))  # process data_inputs iterable with pool
        # # end timer / beendet den Timer
        # toc = time.perf_counter()
        # # print time elapsed / benoetigte Zeit ausgeben
        # print(f"Time elapsed {toc - tic:0.4f} seconds")
        # pool2.close()
        # pool2.join()



