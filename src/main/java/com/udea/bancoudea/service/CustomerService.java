package com.udea.bancoudea.service;

import com.udea.bancoudea.DTO.CustomerDTO;
import com.udea.bancoudea.entity.Customer;
import com.udea.bancoudea.mapper.CustomerMapper;
import com.udea.bancoudea.repository.CustomerRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class CustomerService {

    private final CustomerRepository customerRepository;
    private final CustomerMapper customerMapper;

    @Autowired
    public CustomerService(CustomerRepository customerRepository, CustomerMapper customerMapper) {
        this.customerRepository = customerRepository;
        this.customerMapper = customerMapper;
    }

    public List<CustomerDTO> getAllCustomer(){
        return customerRepository.findAll().stream()
                .map(customerMapper::toDTO).toList();
    }

    public CustomerDTO getCustomerById(Long id){
        return customerRepository.findById(id).map(customerMapper::toDTO)
                .orElseThrow(()->new RuntimeException("Cliente no encontrado"));
    }

    public CustomerDTO createCustomer(CustomerDTO customerDTO){
        Customer customer = customerMapper.toEntity(customerDTO);
        return customerMapper.toDTO(customerRepository.save(customer));
    }

    public CustomerDTO updateCustomer(Long id, CustomerDTO customerDTO){
        Customer existingCustomer = customerRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Cliente no encontrado"));

        if(customerDTO.getAccountNumber() != null){
            existingCustomer.setAccountNumber(customerDTO.getAccountNumber());
        }
        if(customerDTO.getFirstName() != null){
            existingCustomer.setFirstName(customerDTO.getFirstName());
        }
        if(customerDTO.getLastName() != null){
            existingCustomer.setLastName(customerDTO.getLastName());
        }
        if(customerDTO.getBalance() != null){
            existingCustomer.setBalance(customerDTO.getBalance());
        }

        return customerMapper.toDTO(customerRepository.save(existingCustomer));
    }

    public void deleteCustomer(Long id){
        if(!customerRepository.existsById(id)){
            throw new RuntimeException("Cliente no encontrado");
        }
        customerRepository.deleteById(id);
    }

}
