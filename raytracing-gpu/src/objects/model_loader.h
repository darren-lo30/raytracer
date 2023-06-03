#pragma once

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <cassert>
#include <memory>

#include "model.h"
#include <iostream>

namespace ModelLoader {
	Model load_model(std::string path);
	void process_node(aiNode* node, const aiScene* scene);
	Mesh process_mesh(aiMesh* mesh, const aiScene* scene);
	// inline std::vector<Texture> loadMaterialTextures(aiMaterial* material, aiTextureType textureType);

	extern std::string directory;
	extern std::vector<Mesh> meshes;
}