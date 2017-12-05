
#include <boost/program_options.hpp>
#include <fenv.h>
#include <string>
#include <iostream>

#include "Log.hpp"
#include "ModuleList.hpp"
#include "Output.hpp"
#include "Domain.hpp"
#include "InitialCondition.hpp"
#include "SystemSolver.hpp"
#include "Git.hpp"
#include "CUDA.hpp"


int main (int argc, char *argv[])
{
  // Command line options
  boost::program_options::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    ("log-level", boost::program_options::value<std::string>(), "set log level")
    ("set-device", boost::program_options::value<unsigned int>(), "set CUDA device")
    ("disable-cuda", "disable CUDA")
    ("version", "print version")
    ("settings-file", boost::program_options::value<std::string>(), "settings file")
    ;
  boost::program_options::positional_options_description pod;
  pod.add("settings-file", -1);

  boost::program_options::variables_map vm;
  boost::program_options::store(boost::program_options::command_line_parser(argc, argv).
      options(desc).positional(pod).run(), vm);
  boost::program_options::notify(vm);    

  if (vm.count("help"))
  {
    std::cout << desc << "\n";
    return 1;
  }

  if (vm.count("version"))
  {
    std::cout << "Git commit id: " << GIT_ID << std::endl;
#ifdef DEBUG
    std::cout << "Version: Debug" << std::endl;
#else
    std::cout << "Version: Production" << std::endl;
#endif
    return 0;
  }

  // Set logging level
  Log::setLevel(Log::Level::info); // Default level is info
  if (vm.count("log-level"))
  {
    std::string logLevel = vm["log-level"].as<std::string>();
    Log::setLevel(Log::stringToLevel(logLevel));
  }

  if (vm.count("disable-cuda"))
  {
    BOOST_LOG_TRIVIAL(info) << "CUDA is disabled";
  }
  else
  {
    // Set CUDA device
    CUDA::setDevice(CUDA::deviceMostFreeMemory());
    if (vm.count("set-device"))
    {
      const unsigned int device = vm["set-device"].as<unsigned int>();
      CUDA::setDevice(device);
    }
    BOOST_LOG_TRIVIAL(info) << "Running on CUDA device " << CUDA::device();
  }

  std::string settingsFileName;
  if (vm.count("settings-file"))
  {
    settingsFileName = vm["settings-file"].as<std::string>();
  }

  // Initalise from settings file
  Parameters params;
  params.readFile(settingsFileName);
  ModuleList::initialiseFromParameters(params);

  // Enable floating point exceptions
  feenableexcept(FE_DIVBYZERO | FE_OVERFLOW | FE_INVALID);

  // Get active modules
  std::list<std::shared_ptr<Output> > activeOutputModules = ModuleList::mutableModules<Output>();
  std::shared_ptr<const InitialCondition> initialCond = ModuleList::uniqueModule<InitialCondition>();

  // Setup boundary conditions
  std::shared_ptr<BoundaryConditionContainer> bcs = std::make_shared<BoundaryConditionContainer>();
  bcs->initialiseFromParameters(params);

  // Initialise solver
  std::shared_ptr<SystemSolver> solver = std::make_shared<SystemSolver>();
  solver->initialiseFromParameters(params);
  solver->setBoundaryConditions(bcs);

  // Set initial condition 
  const unsigned int nCells = ModuleList::uniqueModule<Domain>()->cells();
  std::shared_ptr<DataPatch> data = std::make_shared<DataPatch>(nCells);
  initialCond->setData(data);

  // Prepare for main loop
  real t=0;
  unsigned int iter=0;

  // Output all data in the beginning
  for(auto outputMod : activeOutputModules)
  {
    outputMod->output(data,t,iter);
  }

  while (t<solver->finalT())
  {
    // Compute time step
    real dt = solver->maxDt(data,t);
    for(auto outputMod : activeOutputModules)
    {
      dt = std::min(dt,outputMod->maxDt(t,dt,iter));
    }

    // Advance solution
    solver->advance(data,t,dt,iter);

    t += dt;
    iter++;
    BOOST_LOG_TRIVIAL(info) << std::setfill('0') << std::setw(5) << iter << std::setprecision(8) << ": t = " << t << " dt = " << dt;

    // Output data if necessary 
    for(auto outputMod : activeOutputModules)
    {
      if(outputMod->needsOutput(t,dt,iter))
      {
        outputMod->output(data,t,iter);
      }
    }
  }

  // Output all data in the end
  for(auto outputMod : activeOutputModules)
  {
    outputMod->output(data,t,iter);
  }

  int exitcode = solver->exitcode();

  BOOST_LOG_TRIVIAL(info) << "Simulation finished.";

  // Tidy up
  ModuleList::clear();

  return exitcode;
}
