package com.udea.bancoudea.service;

import com.udea.bancoudea.dto.CustomerDTO;
import com.udea.bancoudea.entity.Customer;
import com.udea.bancoudea.exception.CustomerNotFoundException;
import com.udea.bancoudea.mapper.CustomerMapper;
import com.udea.bancoudea.repository.CustomerRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

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
                .orElseThrow(() -> new CustomerNotFoundException(id));
    }

    public CustomerDTO createCustomer(CustomerDTO customerDTO){
        if (customerDTO.getBalance() == null) {
            throw new IllegalArgumentException("Balance cannot be null");
        }
        Customer customer = customerMapper.toEntity(customerDTO);
        return customerMapper.toDTO(customerRepository.save(customer));
    }

    public CustomerDTO updateCustomer(Long id, CustomerDTO customerDTO){
        Customer existingCustomer = customerRepository.findById(id)
                .orElseThrow(() -> new CustomerNotFoundException(id));

        Optional.ofNullable(customerDTO.getAccountNumber()).ifPresent(existingCustomer::setAccountNumber);
        Optional.ofNullable(customerDTO.getFirstName()).ifPresent(existingCustomer::setFirstName);
        Optional.ofNullable(customerDTO.getLastName()).ifPresent(existingCustomer::setLastName);
        Optional.ofNullable(customerDTO.getBalance()).ifPresent(existingCustomer::setBalance);

        return customerMapper.toDTO(customerRepository.save(existingCustomer));
    }

    public void deleteCustomer(Long id){
        customerRepository.findById(id).orElseThrow(() -> new CustomerNotFoundException(id));
        customerRepository.deleteById(id);
    }

}
