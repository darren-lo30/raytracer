#pragma once

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <cassert>
#include <memory>

#include "model.h"
#include <iostream>

namespace ModelLoader {
	Model loadModel(std::string path);
	void processNode(aiNode* node, const aiScene* scene);
	Mesh processMesh(aiMesh* mesh, const aiScene* scene);
	// inline std::vector<Texture> loadMaterialTextures(aiMaterial* material, aiTextureType textureType);

	extern std::string directory;
	extern std::vector<Mesh> meshes;
}