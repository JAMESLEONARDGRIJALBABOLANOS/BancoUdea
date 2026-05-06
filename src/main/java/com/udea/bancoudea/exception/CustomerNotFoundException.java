package com.udea.bancoudea.exception;

public class CustomerNotFoundException extends RuntimeException {

    public CustomerNotFoundException(Long id) {
        super("Cliente no encontrado con id: " + id);
    }

    public CustomerNotFoundException(String accountNumber) {
        super("Cliente no encontrado con cuenta: " + accountNumber);
    }
}
