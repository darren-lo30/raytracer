#include "model_loader.h"

namespace ModelLoader {
	std::string directory;
	std::vector<Mesh> meshes;
}

Model ModelLoader::load_model(std::string path) {
	// Import scene
	Assimp::Importer importer;
	const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);

	if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
		// Scene was not loaded in properlyS
		throw importer.GetErrorString();
	}

	// Get prefix of file path
	directory = path.substr(0, path.find_last_of('/'));

	// Start processing nodes recursively
	process_node(scene->mRootNode, scene);

	Model model;
	model.meshes = meshes;	

	return model;
}

void ModelLoader::process_node(aiNode* node, const aiScene* scene) {
	// Node stores indices of meshes stored in the Scene

	for (unsigned int i = 0; i < node->mNumMeshes; i++) {
		// Get index of node's meshes
		unsigned int meshIndex = node->mMeshes[i];

		// Get actual mesh from the index store in the scene
		aiMesh* mesh = scene->mMeshes[meshIndex];
		meshes.push_back(process_mesh(mesh, scene));
	}


	// Recursivly process child nodes
	for (unsigned int i = 0; i < node->mNumChildren; i++) {
		process_node(node->mChildren[i], scene);
	}
}

Mesh ModelLoader::process_mesh(aiMesh* mesh, const aiScene* scene) {
	std::vector<std::shared_ptr<vec3>> vertices;
	std::vector<unsigned int> indices;

	for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
		// Vertex position
		aiVector3D& aiPosition = mesh->mVertices[i];
		std::shared_ptr<vec3> vertex = std::make_shared<vec3>((float) aiPosition.x, (float) aiPosition.y, (float) aiPosition.z);
		vertices.push_back(vertex);
	}
	std::vector<Triangle> mesh_triangles(mesh->mNumFaces);
	for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
		aiFace& face = mesh->mFaces[i];
		assert(face.mNumIndices == 3); // Triangular mesh
		mesh_triangles[i] = Triangle(*vertices[face.mIndices[0]], *vertices[face.mIndices[1]], *vertices[face.mIndices[2]], nullptr);
	}

	Mesh model_mesh;
	model_mesh.triangles = mesh_triangles;
	return model_mesh;		
}
