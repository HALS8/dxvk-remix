#include "d3d9_shader.h"

#include "d3d9_caps.h"
#include "d3d9_device.h"
#include "d3d9_util.h"
#include "../dxvk/dxvk_scoped_annotation.h"
#include "../util/xxHash/xxhash.h"

#include <algorithm>
#include <unordered_set>


namespace dxvk {

  D3D9CommonShader::D3D9CommonShader()
    : m_bytecodeHash(XXH3_64bits(nullptr, 0)) {}

  // Constants are ignored by name because a game assigns registers per shader: the same camera
  // matrix can sit at different registers in different shaders, while its name does not change.
  void D3D9CommonShader::MapFloatConstantNames(const DxsoCtab& ctab) {
    std::unordered_set<std::string> ignoredNames;
    for (const std::string& name : str::split(D3D9Rtx::vertexShaderHashIgnoredConstantNames(), ',')) {
      const size_t first = name.find_first_not_of(" \t");
      const size_t last = name.find_last_not_of(" \t");
      if (first != std::string::npos) {
        ignoredNames.insert(name.substr(first, last - first + 1));
      }
    }

    for (const DxsoCtab::Constant& constant : ctab.m_constantData) {
      if (constant.registerSet != DxsoCtab::registerSetFloat4) {
        continue;
      }
      const uint32_t begin = std::min(constant.registerIndex, caps::MaxFloatConstantsVS);
      const uint32_t end = std::min(constant.registerIndex + constant.registerCount, caps::MaxFloatConstantsVS);
      if (end > m_floatConstantNames.size()) {
        m_floatConstantNames.resize(end);
      }
      for (uint32_t r = begin; r < end; ++r) {
        m_floatConstantNames[r] = constant.name;
      }
      if (begin < end && ignoredNames.count(constant.name) != 0) {
        m_hashIgnoredFloatConstants.emplace_back(begin, end);
      }
    }

    std::sort(m_hashIgnoredFloatConstants.begin(), m_hashIgnoredFloatConstants.end());
    std::vector<std::pair<uint32_t, uint32_t>> merged;
    for (const auto& range : m_hashIgnoredFloatConstants) {
      if (!merged.empty() && range.first <= merged.back().second) {
        merged.back().second = std::max(merged.back().second, range.second);
      } else {
        merged.push_back(range);
      }
    }
    m_hashIgnoredFloatConstants = std::move(merged);
  }

  D3D9CommonShader::D3D9CommonShader(
            D3D9DeviceEx*         pDevice,
            VkShaderStageFlagBits ShaderStage,
      const DxvkShaderKey&        Key,
      const DxsoModuleInfo*       pDxsoModuleInfo,
      const void*                 pShaderBytecode,
      const DxsoAnalysisInfo&     AnalysisInfo,
            DxsoModule*           pModule) {
    const uint32_t bytecodeLength = AnalysisInfo.bytecodeByteLength;
    m_bytecode.resize(bytecodeLength);
    std::memcpy(m_bytecode.data(), pShaderBytecode, bytecodeLength);
    // Same value D3D9Rtx::computeHash computed per shader-capture draw (rtx.cacheShaderBytecodeHash).
    m_bytecodeHash = XXH3_64bits(m_bytecode.data(), m_bytecode.size());

    const std::string name = Key.toString();
    Logger::debug(str::format("Compiling shader ", name));
    
    // If requested by the user, dump both the raw DXBC
    // shader and the compiled SPIR-V module to a file.
    const std::string dumpPath = env::getEnvVar("DXVK_SHADER_DUMP_PATH");
    
    if (dumpPath.size() != 0) {
      DxsoReader reader(
        reinterpret_cast<const char*>(pShaderBytecode));

      reader.store(std::ofstream(str::tows(str::format(dumpPath, "/", name, ".dxso").c_str()).c_str(),
        std::ios_base::binary | std::ios_base::trunc), bytecodeLength);

      char comment[2048];
      Com<ID3DBlob> blob;
      HRESULT hr = DisassembleShader(
        pShaderBytecode,
        TRUE,
        comment, 
        &blob);
      
      if (SUCCEEDED(hr)) {
        std::ofstream disassembledOut(str::tows(str::format(dumpPath, "/", name, ".dxso.dis").c_str()).c_str(), std::ios_base::binary | std::ios_base::trunc);
        disassembledOut.write(
          reinterpret_cast<const char*>(blob->GetBufferPointer()),
          blob->GetBufferSize());
      }
    }
    
    // Decide whether we need to create a pass-through
    // geometry shader for vertex shader stream output

    const D3D9ConstantLayout& constantLayout = ShaderStage == VK_SHADER_STAGE_VERTEX_BIT
      ? pDevice->GetVertexConstantLayout()
      : pDevice->GetPixelConstantLayout();
    m_shaders      = pModule->compile(*pDxsoModuleInfo, name, AnalysisInfo, constantLayout);
    m_isgn         = pModule->isgn();
    // NV-DXVK start: expose shader outputs for vertex capture
    m_osgn = pModule->osgn();
    // NV-DXVK end
    m_usedSamplers = pModule->usedSamplers();

    // Shift up these sampler bits so we can just
    // do an or per-draw in the device.
    // We shift by 17 because 16 ps samplers + 1 dmap (tess)
    if (ShaderStage == VK_SHADER_STAGE_VERTEX_BIT)
      m_usedSamplers <<= caps::MaxTexturesPS + 1;

    m_usedRTs      = pModule->usedRTs();

    m_info      = pModule->info();
    m_meta      = pModule->meta();
    m_constants = pModule->constants();
    m_maxDefinedConst = pModule->maxDefinedConstant();

    if (ShaderStage == VK_SHADER_STAGE_VERTEX_BIT) {
      MapFloatConstantNames(pModule->ctab());
    }

    m_shaders[0]->setShaderKey(Key);

    if (m_shaders[1] != nullptr) {
      // Lets lie about the shader key type for the state cache.
      m_shaders[1]->setShaderKey({ VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, Key.sha1() });
    }
    
    if (dumpPath.size() != 0) {
      std::ofstream dumpStream(
        str::tows(str::format(dumpPath, "/", name, ".spv").c_str()).c_str(),
        std::ios_base::binary | std::ios_base::trunc);
      
      m_shaders[0]->dump(dumpStream);
    }

    pDevice->GetDXVKDevice()->registerShader(m_shaders[0]);

    if (m_shaders[1] != nullptr)
      pDevice->GetDXVKDevice()->registerShader(m_shaders[1]);
  }


  void D3D9ShaderModuleSet::GetShaderModule(
            D3D9DeviceEx*         pDevice,
            D3D9CommonShader*     pShaderModule,
            VkShaderStageFlagBits ShaderStage,
      const DxsoModuleInfo*       pDxbcModuleInfo,
      const void*                 pShaderBytecode) {
    ScopedCpuProfileZone();
    DxsoReader reader(
      reinterpret_cast<const char*>(pShaderBytecode));

    DxsoModule module(reader);

    if (module.info().majorVersion() > pDxbcModuleInfo->options.shaderModel)
      throw DxvkError("GetShaderModule: Out of range of supported shader model");

    if (module.info().shaderStage() != ShaderStage)
      throw DxvkError("GetShaderModule: Bytecode does not match shader stage");

    DxsoAnalysisInfo info = module.analyze();

    DxvkShaderKey lookupKey = DxvkShaderKey(
      ShaderStage,
      Sha1Hash::compute(pShaderBytecode, info.bytecodeByteLength));

    // Use the shader's unique key for the lookup
    { std::unique_lock<dxvk::mutex> lock(m_mutex);
      
      auto entry = m_modules.find(lookupKey);
      if (entry != m_modules.end()) {
        *pShaderModule = entry->second;
        return;
      }
    }
    
    // This shader has not been compiled yet, so we have to create a
    // new module. This takes a while, so we won't lock the structure.
    *pShaderModule = D3D9CommonShader(
      pDevice, ShaderStage, lookupKey,
      pDxbcModuleInfo, pShaderBytecode,
      info, &module);
    
    // Insert the new module into the lookup table. If another thread
    // has compiled the same shader in the meantime, we should return
    // that object instead and discard the newly created module.
    { std::unique_lock<dxvk::mutex> lock(m_mutex);
      
      auto status = m_modules.insert({ lookupKey, *pShaderModule });
      if (!status.second) {
        *pShaderModule = status.first->second;
        return;
      }
    }
  }

}
