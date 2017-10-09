
#ifndef FACTORY_H_
#error FactoryTemplates.cpp must be included in Factory.H
#endif



template<typename T>
std::shared_ptr<T> Factory::create(const std::string& classname)
{
  std::shared_ptr<ParameterisedModuleBase> mod = createBase(classname);

  std::shared_ptr<T> t = std::dynamic_pointer_cast<T>(mod);
  if(!t)
  {
    BOOST_LOG_TRIVIAL(error) << "Could not cast created instance of " << classname << " to the requested class.";
    exit(2);
  }
  if(t->moduleName() != classname)
  {
    BOOST_LOG_TRIVIAL(error)  << "Mismatch of identifiers while creating class. Name method of created class returns " << t->moduleName() << ", however identifier of class is " << classname << ". Make sure the identifier given in the header of file of the class under the 'REGISTER' macro matches the name method.";
    exit(2);
  }

  return t;
}


template<typename T>
std::shared_ptr<T> Factory::createInitialised(const std::string& classname)
{
  std::shared_ptr<T> m = Factory::create<T>(classname);
  m->initialiseFromFile();
  return m;
}
